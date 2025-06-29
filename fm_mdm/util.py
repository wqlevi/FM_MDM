# utility function for diffusion and FM runners

import os
import time
from torch.utils.tensorboard import SummaryWriter
import wandb

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self): 
        return BackgroundGenerator(super().__iter__()) # When yielded, returns BG.__iter__ for faster prefetching

def setup_loader(dataset, batch_size, num_workers=4): 
    #loader = DataLoader(
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    while True:
        yield from loader

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass 
    def add_image(self, step, key, image):
        pass 
    def close(self): pass

class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter,self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt.wandb_api_key)
            wandb.init(dir=str(opt.log_dir), project="FM_celebA256", entity=opt.wandb_user, name=opt.name, config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            # adopt from torchvision.utils.save_image
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter,self).__init__(opt)
        if self.rank == 0:
            run_dir = str(opt.log_dir / opt.name)
            os.makedirs(run_dir, exist_ok=True)
            self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()

def build_log_writer(opt):
    if opt.log_writer == 'wandb': return WandBWriter(opt)
    elif opt.log_writer == 'tensorboard': return TensorBoardWriter(opt)
    else: return BaseWriter(opt) # do nothing

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

# wrapper function to measure time spent in fn
def time_wrap(name:str=None):
    def time_wrap(fn):
        def inner(*args, **kwargs):
            start_time = time.time()
            out = fn(*args, **kwargs)
            print(f"Function {name}\tTime spent : {time.time() - start_time} sec")
            time_used = time.time() - start_time
            return time_used, out
        return inner
    return time_wrap


