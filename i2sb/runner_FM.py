# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle
import psutil

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchdyn.core import NeuralODE
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image128Net
from .network import Image256Net
from .diffusion import Diffusion
from .torchcfm.models.unet import UNetModel
from .torchcfm.conditional_flow_matching import *

from ipdb import set_trace as debug

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.local_rank), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        opt.device = opt.local_rank
        model_arch_dict= {
        'learn_sigma':False,
        'use_checkpoint': False,
        'num_heads': 1,
        'num_head_channels': -1,
        'num_heads_upsample': -1,
        'dropout': 0,
        'resblock_updown': False,
        'use_fp16': False,
        }

        self.net = UNetModel(dim=(3, opt.image_size, opt.image_size),
                             num_channels=128,
                             num_res_blocks=2,
                             **model_arch_dict).to(opt.device)
        
        log.info(f"[Net] Net work size={util.count_parameters(self.net)}!")

        self.node = NeuralODE(self.net, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        log.info(f"[Flow matching] Built FM!")

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")

        self.net.to(opt.device)

        self.log = log

    @util.time_wrap("sample_time")
    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader) # type(loader) = 'generator' 
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device]) # device_ids = [int]
        #net = DDP(self.net, device_ids=[opt.local_rank]) # device_ids = [int]
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch, num_workers=0) # len(train_dataset) = 1281167
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch, num_workers=0)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        for it in range(opt.num_itr):
            optimizer.zero_grad()
            runtime = 0  
            for _ in range(n_inner_loop): # only cumulate gradients here:
                # ===== sample boundary pair =====
                time, (x0, x1, mask, y, cond) = self.sample_batch(opt, train_loader, corrupt_method) 
                t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
                # ===== compute loss =====

                vt = net(t, xt)

                loss = F.mse_loss(vt, ut)
                loss.backward()
                
                runtime += float(time)#[TODO]
            optimizer.step()
            runtime /= n_inner_loop
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{} | runtime ave:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
                "{:+.3f}".format(runtime),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 500 == 0 and it != 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier() # wait till all GPU complete the iteration and synchronize 

            if it == 100 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def FM_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        traj = self.node.trajectory(x1, t_span = torch.linspace(0,1,2).to(opt.device)) # traj: [t_span, B, ...]
        traj = traj.permute(1,0,2,3,4)
        return traj 

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        _,(img_clean, img_corrupt, mask, y, cond) = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)
        xs = self.FM_sampling(opt, x1) # xs:[B, 2, xdim]

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs) # [B, 2, ...]
        #pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim), "got img_clean shape:{}, img_corrupt shape: {}".format(img_clean.shape, img_corrupt.shape)
        #assert xs.shape == pred_x0s.shape
        assert y.shape == (batch,)
        #log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]


        log.info("Logging images ...")
        img_recon = xs[:,0]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        #log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        #log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
