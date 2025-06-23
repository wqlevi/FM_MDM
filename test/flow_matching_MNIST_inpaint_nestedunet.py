# TODO
# - [x] OT coupling before sampling mini-batch, for unpaired translation
# - [ ] Using same flow matching loss as I2SB eq12.
# - [x] resized MNIST, step=1 training
# - [ ] model is not robust for different num_channles (eg. ERROR at num_channel=128, num_res_block=2)
# - [x] need implementation for multi-resolution input simultaneously, and calculate summed FM loss

#!/usr/bin/env python
import math
import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
from pudb.remote import set_trace as debug

from torch import Tensor
from tqdm import tqdm
from typing import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, AdamW
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, Resize
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchdyn.core import NeuralODE
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.multiprocessing import spawn
from prefetch_generator import BackgroundGenerator
import wandb

from dataset.LMDB2ImageFolder import Dset
#from torchcfm.models.unet import NestedUNetModel, UNetModel
from fm_mdm.torchcfm.models.unet import nestedunet
from fm_mdm.torchcfm.conditional_flow_matching import *
from build_mask import build_inpaint_center
import distributed_util as dist_util

def seed_everything(seed:int=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

def count_params(model):
    num = sum(para.data.nelement() for para in model.parameters())
    count = num / 1024**2
    print(f"Model num params: {count=:.2f} M")

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_optimizer_sched(net):
    lr_gamma = 0.99
    lr_step = 1000
    optim_dict = {"lr": 5e-5, "weight_decay":0.0}
    optimizer = AdamW(net.parameters(), **optim_dict)
    sched_dict = {"step_size": lr_step, "gamma": lr_gamma}
    sched = lr_scheduler.StepLR(optimizer, **sched_dict)
    return optimizer, sched

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1233'
    init_process_group("nccl", rank=rank, world_size=world_size)

def spawn_fn(fn):
    world_size = 4
    spawn(fn,
          args= (world_size,),
          nprocs = world_size,
          join = True)

def count_param(model):
    if isinstance(model, torch.nn.Module):
        count = sum(para.data.nelement() for para in model.parameters())
        count /= 1024**2
        print(f"Num of params: {count=:.2f} M")

def sample_batch(rank, clean_img, y, corrupt_method, resize:int=8):
    #clean_img, y = next(loader)
    clean_img  = Resize(resize)(clean_img) 
    with torch.no_grad():
        corrupt_img, mask = corrupt_method(clean_img.to(rank))

    y  = y.detach().to(rank)
    x0 = clean_img.detach().to(rank)
    x1 = corrupt_img.detach().to(rank)
    if mask is not None:
        mask = mask.detach().to(rank)
        x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
    cond = None
    assert x0.shape == x1.shape
    return x0, x1, mask, y, cond

def all_cat_cpu(x_ls:List[Tensor], rank)-> Tensor:
    x = x_ls[0]
    gathered_x = dist_util.all_gather(x.to(rank)) 
    return torch.cat(gathered_x).detach().cpu()

def build_loader(rank, world_size, dataset, batch_size):
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoaderX(dataset, batch_size = batch_size, sampler = data_sampler, drop_last=True)
    while True:
        yield from dataloader



def main(rank, world_size, *args, **kwargs):
    seed_everything(2024)
    init_process(rank, world_size)

    global IMG_SIZE , LOG_FLAG
    LOG_FLAG = False
    IMG_SIZE = 256
    global_size = world_size

    if LOG_FLAG and rank == 0:
        wandb.init(project='FM_celebA256')
    corrupt_method_32 = build_inpaint_center(IMG_SIZE, rank)
    if rank == 0: os.makedirs("./results/UNet", exist_ok=True)

    ITERS= 1
    #ITERS=610
    #batch_size= 256
    dataset_dir = '/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/'
    batch_size= 256
    microbatch = 8
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

    model = nestedunet().to(rank) 
    #model = NestedUNetModel(dim=(1, IMG_SIZE, IMG_SIZE), num_channels=32, num_res_blocks=1, channel_mult=(2,2,2), **model_arch_dict).to(rank) 
    count_param(model)
    node = NeuralODE(model, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
    model = DDP(model, device_ids=[rank])

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer, sched = build_optimizer_sched(model)

    n_inner_loop = batch_size // (global_size * microbatch)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    #---------source dataset----------#
    data_src = Dset(Path(dataset_dir, "train", "subclass_train"), IMG_SIZE)
    dataloader = build_loader(rank, world_size, data_src, microbatch)

    corrupt_method_1 = build_inpaint_center(int(IMG_SIZE//4), rank) # 64x64
    corrupt_method_2 = build_inpaint_center(int(IMG_SIZE//2), rank) # 128x128
    corrupt_method_3 = build_inpaint_center(int(IMG_SIZE), rank)    # 256x256
    #corrupt_method_all_res = [ corrupt_method_32]
    #res_group = [IMG_SIZE]

    """
    ckpt = torch.load("./results/debug_runtime/latest.pt")
    model.load_state_dict(ckpt['net'])
    """
    model.train()
    for it in range(ITERS):
        #print(f"\033[1m*****CURRENT {epoch=}*****\033[0m")
        if it < ITERS//3:
            step=0
            corrupt_method_all_res = [corrupt_method_1,]
            res_group = [int(IMG_SIZE// 2**2),]
        elif (it > ITERS//3) and (it< ITERS*2//3):
            step=1
            corrupt_method_all_res = [corrupt_method_1, corrupt_method_2,]
            res_group = [int(IMG_SIZE// 2**2), int(IMG_SIZE)] # 64x64, 256x256
        else:
            step=2
            corrupt_method_all_res = [corrupt_method_1, corrupt_method_2, corrupt_method_3]
            res_group = [int(IMG_SIZE// 2**2), int(IMG_SIZE//2), int(IMG_SIZE)] # 64x64, 256x256

        optimizer.zero_grad()
        # ---------- step configs ---------- # 
        for _ in range(n_inner_loop):
            x0_l, x1_l = [], []
            clean_img, y = next(dataloader) 
            tmp_ls = [sample_batch(rank, clean_img, y, corrupt_method, res) for corrupt_method, res in zip( corrupt_method_all_res, res_group)]
            res_ls = [FM.sample_location_and_conditional_flow(tmp[1], tmp[0]) for tmp in tmp_ls] # x0: clean, x1: paint
            t = res_ls[0][0]
            xt_ls = [sub[1] for sub in res_ls]
            ut_ls = [sub[2] for sub in res_ls]
            print(f"{type(xt_ls)} and {len(xt_ls)} and inner type {type(xt_ls[0])}")
            vt_ls = model(t, *xt_ls) # t: [256]; xt: [256,1,28,28]
            loss_v = torch.stack([F.mse_loss(vt, ut) for vt, ut in zip(vt_ls, ut_ls)]).mean()
            loss_v.backward()
            loss_value = loss_v.item()
        optimizer.step()
        if sched is not None: sched.step()


        with torch.no_grad():
            if rank==0: print("train_it {}/{} | lr: {} | loss: {}".format(1+it, ITERS, "{:.2e}".format(optimizer.param_groups[0]['lr']), "{:.4f}".format(loss_value)))
            if LOG_FLAG and rank==0: wandb.log({'loss':loss_value})

        if it % 100 == 0:
            evaluate(rank, dataloader, node, corrupt_method_all_res, res_group, it)
        if it % 300 == 0 and rank==0:
            torch.save(model.state_dict(),"./results/UNet/celeba_{}.ckpt".format(it))
    barrier()
    destroy_process_group()
    # -------- sampling ----------#

@torch.no_grad()
def evaluate(rank, dataloader, node, corrupt_method_all_res, res_group, it):
    print(f"====== Evaluation started: iter={it} =====")
    clean_img, y = next(dataloader) 
    val_ls = [sample_batch(rank, clean_img, y, corrupt_method, res) for corrupt_method, res in zip( corrupt_method_all_res, res_group)]

    x1_ls = [x[1] for x in val_ls] # list of x1 tensor: [B, C, H,W]
    x0_ls = [x[0] for x in val_ls] # list of x0 tensor: [B, C, H,W]

    traj = [node.trajectory(x1, t_span = torch.linspace(0,1,2).to(rank)) for x1 in x1_ls]# dim of traj?

    traj_end = [j[-1] for j in traj]

    x0_ls    = all_cat_cpu(x0_ls, rank) # [B*GPUs, ...]
    x1_ls    = all_cat_cpu(x1_ls, rank)
    traj_end = all_cat_cpu(traj_end, rank) # List
    #debug(term_size=(100,30))

    def plot_result(x:List[Tensor], caption:str=None):
        titles = ['x0', 'x1', 'x0_ode'] 
        grids = [make_grid(xi[:100].clip(-1, 1),value_range=(-1,1), padding=0, nrow=10) for xi in x] 
        if LOG_FLAG and rank == 0: wandb.log({caption: [wandb.Image(xi, caption=txt) for xi, txt in zip(grids, titles)]})

    os.makedirs("./results", exist_ok=True)
    x0_ls, x1_ls, traj_end=[[x,] for x in [x0_ls, x1_ls, traj_end] if not isinstance(x, list)] 
    [plot_result(x, "res_{}".format(2**(3+i))) for i, x in enumerate(zip(x0_ls, x1_ls, traj_end))]

#model.load_state_dict(torch.load("./checkpoints/nestedUNet/debug.ckpt"))


if __name__ == '__main__':

    spawn_fn(main)

