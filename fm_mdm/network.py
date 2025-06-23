# defining network architecture based on given configurations.

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG128_UNCOND_PKL,
    I2SB_IMG256_SMALL_UNCOND_PKL,
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

from ipdb import set_trace as debug
class Image128Net(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=False, ckpt_dir="data/"):
        super(Image128Net, self).__init__()

        # initialize model
        #ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG128_COND_PKL if cond else I2SB_IMG128_UNCOND_PKL) # 128x128
        ckpt_pkl = os.path.join(ckpt_dir,I2SB_IMG256_SMALL_UNCOND_PKL)  # 256x256 small network
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16 # replacing fp16 to fp32 as default
        self.diffusion_model = create_model(**kwargs)
        if kwargs["use_fp16"]: 
            self.diffusion_model.convert_to_fp16() 
            log.info(f"[Net] Initialized network from {ckpt_pkl=}! In FP16!")
        log.info(f"[Net] Initialized network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG128_COND_CKPT if cond else I2SB_IMG128_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt=}!")

        self.diffusion_model.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach().type_as(x)
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)

class Image256Net(torch.nn.Module):
    def __init__(self, log, use_fp16=False, cond=False, pretrained_adm=True, ckpt_dir="data/"):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_SMALL_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        self.model = create_model(**kwargs)
        log.info(f"[Net] Initialized network from {ckpt_pkl=}! Size={util.count_parameters(self.model)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt=}!")

        self.model.eval()
        self.cond = cond

    def forward(self, t, x, cond=None):

        return self.model(t, x)
