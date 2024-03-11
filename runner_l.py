import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

from line_profiler import profile

import torch
import torch.nn.functional as F
import lightning as L
from corruption import build_corruption
from dataset import imagenet 
from dataset.LMDB2ImageFolder import Dset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from i2sb.network import Image128Net
from i2sb.diffusion import Diffusion
from i2sb.runner import make_beta_schedule, build_optimizer_sched
from i2sb import util 
from evaluation import build_resnet50
from logger import Logger


def update_ema(model_ema, model_net, decay=0.9999):
    param_ema = dict(model_ema.named_parameters())
    param_net = dict(model_net.named_parameters())
    for k in param_ema.keys():
        param_ema[k].data.mul_(decay).add_(param_net[k].data, alpha=1 - decay)
class Runner(L.LightningModule):
    def __init__(self, log, opt, save_opt=True):
        super().__init__()

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))


        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2],
                           np.flip(betas[:opt.interval//2])])
        self.opt = opt
        self.syslog = log

        self.train_dataset = Dset(Path(opt.dataset_dir, "train", "subclass_train"), opt.image_size)
        self.val_dataset = Dset(Path(opt.dataset_dir, "val", "subcalss_val"), opt.image_size)

        self.diffusion = Diffusion(betas) #TODO no 'device' arg here
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval)

        self.net = Image128Net(log, noise_levels=noise_levels, use_fp16=self.opt.use_fp16, cond=opt.cond_x1)
        #self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.ema = Image128Net(log, noise_levels=noise_levels, use_fp16=self.opt.use_fp16, cond=opt.cond_x1)

        self.resnet = build_resnet50()
        self.corrupt_method = build_corruption(self.opt, log)

        self.backprop_frequency = opt.batch_size // opt.microbatch // opt.n_gpu_per_node
        self.automatic_optimization = False

        if self.opt.load:
            checkpoint = torch.load(self.opt.load, map_location='cpu')
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint['ema'])
            log.info(f"[Net] Loaded ema ckpt: {opt.load}!")

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size= self.opt.microbatch,
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True,
                                  drop_last=True)
        return train_loader

    def training_step(self, batch, batch_idx): 
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x0, x1, mask, y, cond = self.sample_batch(batch, self.corrupt_method)
        step = torch.randint(0, self.opt.interval, (x0.shape[0], ))
        xt = self.diffusion.q_sample(step, x0, x1, ot_ode = self.opt.ot_ode)
        label = self.compute_label(step, x0, x1)
        pred = self.net(xt, step, cond=cond)

        if mask is not None:
            pred = mask*pred
            label = mask*label

        loss = F.mse_loss(pred, label)
        self.manual_backward(loss)
        if (batch_idx % self.backprop_frequency) == 0:
            optimizer.step()
            update_ema(self.ema, self.net)
        # if sched is not None: sched.step() 
        # --- logging and ckp saving --- #
        if batch_idx % 10 == 0:
            self.log("loss", loss.detach())

    def validation_step(self, batch, batch_idx):
        log = self.log
        log.info(f"========== Evaluation started: iter={batch_idx} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(batch, self.corrupt_method)

        x1 = img_corrupt

        xs, pred_x0s = self.ddpm_sampling(
            x1, mask=mask, cond=cond, clip_denoise=self.opt.clip_denoise, verbose=self.opt.global_rank==0
        )
        del x1

        log.info("Collecting tensors ...")
        img_clean   = self.all_gather( img_clean).detach().cpu()
        img_corrupt = self.all_gather( img_corrupt).detach().cpu()
        y           = self.all_gather( y).detach().cpu()
        xs          = self.all_gather( xs).detach().cpu()
        pred_x0s    = self.all_gather( pred_x0s).detach().cpu()

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.logger.log_image(tag, make_grid((img + 1)/2 , nrow=nrow))


        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)


        log.info(f"========== Evaluation finished: iter={self.global_step} ==========")


    def ddpm_sampling(self, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):
        nfe = nfe or self.opt.interval-1
        assert 0 < nfe < self.opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(self.opt.interval, nfe+1)

        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.syslog.info(f"[DDPM Sampling] steps={self.opt.interval}, {nfe=}, {log_steps=}!")

        #x1 = x1.to(opt.device)
        if cond is not None: cond = cond.type_as(x1)
        if mask is not None:
            mask = mask.type_as(x1)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, dtype=torch.long).type_as(x1) #FIXME: move tensor to X1 device
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=self.opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    def validation_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                  batch_size= self.opt.microbatch,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
        return val_loader

    def configure_optimizers(self):
        optimizer, sched = build_optimizer_sched(self.opt, self.net, self.syslog) # FIXME how to make sched in L?
        return optimizer
    # TODO does L has its own setup for sampling?
    def sample_batch(self, batch, corrupt_method):
        if self.opt.corrupt == 'mixture':
            clean_img, corrupt_img, y = batch
            mask = None
        elif self.opt.corrupt == 'inpaint':
            clean_img, y = batch
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img)
        else:
            clean_img, y = batch
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img)
                mask = None
        y = y.detach().to(self.device)
        x0 = clean_img.detach().to(self.device)
        x1 = corrupt_img.detach().to(self.device)
        if mask is not None:
            mask = mask.detach().to(self.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if self.opt.cond_x1 else None
        if self.opt.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)
        return x0, x1, mask, y, cond

    def compute_label(self, step, x0, xt):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        std_fwd = std_fwd.type_as(x0)
        label = (xt - x0) / std_fwd# FIXME GPU and cpu
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_noise=False):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        std_fwd = std_fwd.type_as(xt)
        pred_x0 = xt - std_fwd * net_out
        if clip_noise: pred_x0.clap_(-1., 1.)
        return pred_x0



def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          action="store_true",   default=True)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default="test",        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--microbatch",     type=int,   default=2,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")

    opt = parser.parse_args()
    RESULT_DIR = Path("results")
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)
    
    opt.use_fp16 = False
    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None
    return opt
        
if __name__ == '__main__':
    opt = create_training_options()
    log = Logger(log_dir=opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    diffusion_model = Runner(log, opt)
    wandb_logger = WandbLogger(project = opt.name,
                               log_model = False,
                               group = opt.ckpt)

    bar = TQDMProgressBar(refresh_rate=diffusion_model.backprop_frequency)
    if opt.train:
        checkpoint_callback = ModelCheckpoint(dirpath= opt.ckpt_path,
                                              save_last=True,
                                              save_top_k= -1,
                                              )
        trainer = L.Trainer(accelerator='auto',
                            devices = opt.n_gpu_per_node,
                            num_nodes= opt.num_proc_node,
                            max_steps= opt.num_itr, # this refers to optimizer steps
                            callbacks=[checkpoint_callback, bar],
                            logger= wandb_logger,
                            strategy='ddp_find_unused_parameters_true',
                            val_check_interval=0.1, # FIXME: not validating
                            #fast_dev_run=True,
                            )
        trainer.fit(diffusion_model, 
                    )
