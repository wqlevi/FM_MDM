# ===========
#   main training script
# ===========

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
#from torch.multiprocessing import Process, spawn, set_start_method
from torch.distributed import init_process_group, barrier, destroy_process_group

from logger import Logger
from corruption import build_corruption
from dataset import imagenet
#from i2sb import Runner_FM_MDM as Runner
from fm_mdm import download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default="test",        help="experiment ID")
    parser.add_argument("--train-mode",     type=str,   default="MDM",        help="MDM, MDM256x256, progressive")
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
    parser.add_argument("--use-weighting",  action="store_true",             help="use coeff weighting")
    #parser.add_argument("--use-prounet",  action="store_true",             help="use progressive unet")
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

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    rank = int(os.environ.get("SLURM_PROCID"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    opt.global_rank = rank
    opt.local_rank = rank - opt.n_gpu_per_node * (rank// opt.n_gpu_per_node)
    node_rank = int(os.environ.get("SLURM_NODEID"))
    print(f"main function of global rank {opt.global_rank}\tlocal rank {opt.local_rank}\tnode index {node_rank}\tlocal world size {world_size} of total world size of {opt.global_size}")

    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Flow Matching at various scales")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    init_process_group("nccl", rank=rank, world_size=opt.global_size)
    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build imagenet dataset
    train_dataset = imagenet.build_lmdb_dataset(opt, log, train=True)
    val_dataset   = imagenet.build_lmdb_dataset(opt, log, train=False)
    # note: images should be normalized to [-1,1] for corruption methods to work properly

    if opt.corrupt == "mixture":
        import corruption.mixture as mix
        train_dataset = mix.MixtureCorruptDatasetTrain(opt, train_dataset)
        val_dataset = mix.MixtureCorruptDatasetVal(opt, val_dataset)

    # build corruption method
    corrupt_method = build_corruption(opt, log)

    # parsed train_mode option for different training schemes
    # TODO: convert module selection into LUT
    if opt.train_mode == 'MDM':
        Runner = import_module('fm_mdm').Runner_FM_MDM
    elif opt.train_mode == 'MDM256x256':
        Runner = import_module('fm_mdm').Runner_FM_256x256
    elif opt.train_mode == 'progressive':
        Runner = import_module('fm_mdm').Runner_FM_pro
    else:
        raise NotImplementedError('Train mode unknown')
    run_name = opt.train_mode
    log.info('[Runner]: Using {} strategy!'.format(run_name))
    run = Runner(opt, log) #[FIXED][BYPASSED] using spawn
    run.train(opt, train_dataset, val_dataset, corrupt_method) 
    log.info("Finish!")
    barrier()
    destroy_process_group()


if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    opt.global_size = opt.num_proc_node * opt.n_gpu_per_node
    #main(opt)
    main(opt)
