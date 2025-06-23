# TODO:
# -[x] input always [64x64, 128x128, 256x256]; made during foward call in nn.Module
# -[x] loss and output only for [256x256];


import os
import numpy as np
import pickle
import psutil

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List

from torchdyn.core import NeuralODE
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
from torchvision.transforms import Resize
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .diffusion import Diffusion
from .torchcfm.models.unet import nestedUnet
from .torchcfm.conditional_flow_matching import *
from build_mask import build_inpaint_center

#from ipdb import set_trace as debug

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    # FIXME lr to constant 5e-5
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

def all_cat_cpu(opt, log, t:torch.Tensor):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.local_rank), log=log)# return tensor gathered from all devices
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
        print(f'\033[92mCurrent device-Runner : {torch.cuda.current_device()}\t {opt.device=}\033[0m')

        self.net = nestedUnet(opt.device, min_size=64, training_mode = opt.train_mode) 
        log.info(f"[Net] Net work size={util.count_parameters(self.net)}!")

        self.node = NeuralODE(self.net, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        self.current_iter = 0
        log.info(f"[Flow matching] Built FM!")

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!") 
            self.current_iter= checkpoint['sched']['last_epoch'] if checkpoint['sched'] is not None else checkpoint['epoch']# FIXME conflict with constant LR
            log.info(f"[Iter] Loaded iter ckpt: {self.current_iter}!")

        self.net.to(opt.device)

        self.log = log

    def sample_batch(self, opt, clean_img, corrupt_method, resize:int):
        #clean_img, y = next(loader)
        clean_img = Resize(resize)(clean_img)
        rank = opt.device
        with torch.no_grad():
            corrupt_img, mask = corrupt_method(clean_img.to(rank))

        #y  = y.detach().to(rank)
        x0 = clean_img.detach().to(rank)
        x1 = corrupt_img.detach().to(rank)
        if mask is not None:
            mask = mask.detach().to(rank)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = None
        assert x0.shape == x1.shape
        return x0, x1, mask, cond

    def create_res_group(self, opt, itr):
        # return List[callbacks] List[int]
        return [ build_inpaint_center(opt.image_size//2**i, opt.device) for i in range(3)][::-1], [opt.image_size//2**i for i in range(3)][::-1]

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        coeff_dict = {1:[1,0,0], 2:[.5,1,0], 3:[.25, .5, 1]}
        def get_weight_coeff(len_res:int, coeff_dict:dict):
            return coeff_dict[len_res]
            
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device], find_unused_parameters=True) # device_ids = [int]
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch, num_workers=0) 
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch, num_workers=0)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        for it in range(self.current_iter, opt.num_itr):
            loss_v = 0

            corrupt_method_all_res, res_group = self.create_res_group(opt, it)
            optimizer.zero_grad()
            for _ in range(n_inner_loop): # only cumulate gradients here:
                # ===== sample boundary pair =====
                clean_img, _ = next(train_loader)
                tmp_ls = [self.sample_batch(opt, clean_img, corrupt_method, res) for corrupt_method, res in zip(corrupt_method_all_res, res_group)]
                res_ls = [self.FM.sample_location_and_conditional_flow(tmp[1], tmp[0]) for tmp in tmp_ls]

                # ===== compute loss =====
                xt_ls = [sub_ls[1] for sub_ls in res_ls]
                t = res_ls[0][0]
                ut_ls = [sub_ls[2] for sub_ls in res_ls]
                vt_ls = net(t, xt_ls)

                loss = F.mse_loss(vt_ls[-1], ut_ls[-1])
                loss.backward()
                loss_v += loss.item()
                
            optimizer.step()
            loss_v /= n_inner_loop

            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{} ".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss_v),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss_v)

            if it % 500 == 0 and it != 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                        "epoch": it,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier() # wait till all GPU complete the iteration and synchronize 

            if it == 100 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method_all_res, res_group)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def FM_sampling(self, opt, x1_ls:list, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True): #FIXME: debug inference

        traj = [self.node.trajectory(x1_ls[-1], t_span = torch.linspace(0,1,2).to(opt.device))] # traj: [t_span, B, ...]
        traj_end = [j[-1] for j in traj]
        return traj_end

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method_all_res, res_group):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")
        clean_img, _ = next(val_loader)

        val_ls = [self.sample_batch(opt, clean_img, corrupt_method, res) for corrupt_method, res in zip(corrupt_method_all_res, res_group)]

        res_now = int(res_group[0])
        x1_ls = [x[1] for x in val_ls] # img_ls clean
        x0_ls = [x[0] for x in val_ls] # img_ls corrupt

        traj_ls = self.FM_sampling(opt, x1_ls) # img_ls recon

        log.info("Collecting tensors ...")

        img_ls_clean   = [all_cat_cpu(opt, log, img_clean) for img_clean in x0_ls]
        img_ls_corrupt = [all_cat_cpu(opt, log, img_corrupt) for img_corrupt in x1_ls]
        img_ls_recon   = [all_cat_cpu(opt, log, xs) for xs in traj_ls]# [B, 2, ...]
        #pred_x0s    = all_cat_cpu(opt, log, pred_x0s)
        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]


        log.info("Logging images ...")
        log_image("image/clean_{}".format(res_now),   img_ls_clean[-1])
        log_image("image/corrupt_{}".format(res_now), img_ls_corrupt[-1])
        log_image("image/recon_{}".format(res_now),   img_ls_recon[0])

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
