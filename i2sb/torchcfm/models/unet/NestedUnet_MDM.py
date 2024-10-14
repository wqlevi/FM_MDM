import torch
import torch.nn as nn
from torch import Tensor
from typing import List
import numpy as np

from torchvision.transforms import Resize

from . import UNetModel
from .nn import timestep_embedding
class nestedUnet(nn.Module):
    def __init__(self, rank, **kwargs):
        super().__init__()
        self.net_inner =  UNetModel(dim=(3,64,64), num_channels=128, attention_resolutions="128, 64", channel_mult=[2,4,6] , num_heads=8, num_head_channels=64, num_res_blocks=2).to(rank) # config at I2SB
        self.net_mid =  UNetModel(dim=(3,128,128), num_channels=128, attention_resolutions="64", channel_mult=[1,2] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid=True).to(rank) # config at I2SB
        #self.net_outer =  UNetModel(dim=(3,256,256), num_channels=64, attention_resolutions="2", channel_mult=[1,2,4] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid = True).to("cuda:0") # config at MDM-S64S256
        #self.net_outer =  UNetModel(dim=(3,256,256), num_channels=64, attention_resolutions="2", channel_mult=[1,2] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid = True).to(rank) # config at MDM-S64S128S256
        self.net_outer =  UNetModel(dim=(3,256,256), num_channels=128, attention_resolutions="128", channel_mult=[1,1] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid = True).to(rank)  # config at MDM-S64S128S256 BIG
        self.all_layers = [self.net_inner, self.net_mid, self.net_outer]
        self.dtype = torch.float32
        self.min_size = 32 if not 'min_size' in kwargs else kwargs['min_size']
        #self.use_prounet = False if not 'use_prounet' in kwargs else kwargs['use_prounet']
        self.training_mode = kwargs['training_mode'] if 'training_mode' in kwargs else None

    def _inference_downsample(self, x:Tensor, min_size:int=64)->List[Tensor]:
        """Get list of resized/ downsampled tensors including input tensor itself"""
        factor = int(np.log2(min_size))
        ls_down = [Resize(2**(i+factor))(x) for i in range(int(np.log2(x.shape[-1]//min_size)))]
        return [*ls_down, x]

    def hook_fn(self, m, i, o):
        block_res = f"\n\033[96mlayer_res_{str(i[0].shape[-1])}_\033[0m"
        print(block_res)
        print(f"\033[93minput tensor shape: {i[0].shape}\033[0m")
        print(f"\033[0mID of layer: {id(m)}\033[0m")
        print(f"\033[91moutput tensor shape: {o.shape}\033[0m")

    def f_down(self,h,t,layers, hs)->Tensor:
        for module in layers:
            #module.register_forward_hook(self.hook_fn)
            h = module(h,t)
            hs.append(h)
        return h # return last tensor

    def f_up(self,h,t,layers, hs)->Tensor:
        for module in layers:
            h = torch.cat([h, hs.pop()], dim=1)
            #module.register_forward_hook(self.hook_fn)
            h = module(h, t)
        return h

    def f_mid(self,x,t)-> Tensor:
        return self.net_inner.middle_block(x,t)

    def _forward_prounet(self, z:Tensor, emb:List[Tensor], o:List[Tensor] = []):
        if z.shape[-1] == self.min_size:
            h_64 = self.all_layers[0].in_layer(z, emb[0])
            hs_64 = [h_64]
            h_64 = self.f_up(self.f_mid(self.f_down(h_64, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            ho_64 = self.all_layers[0].out(h_64)
            o.append(ho_64)
        elif z.shape[-1] == self.min_size *2: # 128x128
            h_128 = self.all_layers[1].in_layer(z, emb[1]) # [B,3,HDM] -> [B, C, HDM] 
            hs_128 = [h_128]
            h_128 = self.f_down(h_128, emb[1], self.all_layers[1].input_blocks, hs_128) # [B, C, HDM] -> [B, 2*C, HDM/2] 64x64 here
            hs_64 = [h_128]
            h_64 = self.f_up(self.f_mid(self.f_down(h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]), # [B,2*C, HDM/2] + []
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            h_128 = self.f_up(h_128 + h_64, emb[1], self.all_layers[1].output_blocks, hs_128)
            ho_128 = self.all_layers[1].out(h_128)
            o.append(ho_128)
        elif z.shape[-1] == self.min_size *(2**2):
            h_256 = self.all_layers[2].in_layer(z, emb[2])
            hs_256= [h_256]
            h_256 = self.f_down(h_256, emb[2], self.all_layers[2].input_blocks, hs_256)
            hs_128 = [h_256]
            h_128 = self.f_down(h_256, emb[1], self.all_layers[1].input_blocks, hs_128)
            hs_64 = [h_128]
            h_64 = self.f_up(self.f_mid(self.f_down(h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            h_128 = self.f_up(h_128 + h_64, emb[1], self.all_layers[1].output_blocks, hs_128)
            h_256 = self.f_up(h_256 + h_128, emb[2], self.all_layers[2].output_blocks, hs_256)
            ho_256 = self.all_layers[2].out(h_256)
            o.append(ho_256)

    def _forward_FM256x256Unet(self, z:List[Tensor], emb:List[Tensor], o:List[Tensor]=[]):
        # input: [64x64, 128x128, 256x256]
        # output: [256x256]
        assert len(z) == 3, "input list must contains 3 resolutions!"
        h_64, h_128, h_256 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(3)]
        hs_256, hs_128, hs_64 = [h_256], [h_128], [h_64]
        h_256 = self.f_down(h_256, emb[2], self.all_layers[2].input_blocks, hs_256)
        h_128 = self.f_down(h_128+h_256, emb[1], self.all_layers[1].input_blocks, hs_128) # FIXME: h_128[N, 128, 128, 128]; h_256[N, 256, 128, 128]
        h_64 = self.f_up(self.f_mid(self.f_down(h_64+h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                      emb[0],
                      self.all_layers[0].output_blocks,
                      hs_64)
        #ho_64 = self.all_layers[0].out(h_64)
        #o.append(h_64)
        h_128 = self.f_up(h_128 + h_64, emb[1], self.all_layers[1].output_blocks, hs_128)
        #ho_128 = self.all_layers[1].out(h_128)
        #o.append(h_128)
        h_256 = self.f_up(h_256 + h_128, emb[2], self.all_layers[2].output_blocks, hs_256)
        ho_256 = self.all_layers[2].out(h_256)
        o.append(ho_256)

    def _forward_nestedunet(self, z:List[Tensor], emb:List[Tensor], o:List[Tensor]=[]):
        if  len(z)==1:
            h_64 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(1)][0]
            hs_64 = [h_64]
            h_64 = self.f_up(self.f_mid(self.f_down(h_64, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            ho_64 = self.all_layers[0].out(h_64)
            o.append(ho_64)
        elif len(z)==2:
            h_64, h_128 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(2)]
            hs_128, hs_64 = [h_128], [h_64]
            h_128 = self.f_down(h_128, emb[1], self.all_layers[1].input_blocks, hs_128)
            h_64 = self.f_up(self.f_mid(self.f_down(h_64+h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            ho_64 = self.all_layers[0].out(h_64)
            o.append(ho_64) # final output img
            h_128 = self.f_up(h_128 + h_64, emb[1], self.all_layers[1].output_blocks, hs_128)
            ho_128 = self.all_layers[1].out(h_128)
            o.append(ho_128)
        elif len(z)==3:
            h_64, h_128, h_256 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(3)]
            hs_256, hs_128, hs_64 = [h_256], [h_128], [h_64]
            h_256 = self.f_down(h_256, emb[2], self.all_layers[2].input_blocks, hs_256)
            h_128 = self.f_down(h_128+h_256, emb[1], self.all_layers[1].input_blocks, hs_128) # FIXME: h_128[N, 128, 128, 128]; h_256[N, 256, 128, 128]
            h_64 = self.f_up(self.f_mid(self.f_down(h_64+h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            ho_64 = self.all_layers[0].out(h_64)
            o.append(ho_64)
            h_128 = self.f_up(h_128+h_64, emb[1], self.all_layers[1].output_blocks, hs_128)
            ho_128 = self.all_layers[1].out(h_128)
            o.append(ho_128)
            h_256 = self.f_up(h_256+h_128, emb[2], self.all_layers[2].output_blocks, hs_256)
            ho_256 = self.all_layers[2].out(h_256)
            o.append(ho_256)
        else:
            raise NotImplementedError("Only up to 3 resolution implemented, but size {} got".format(len(z)))
    def forward(self, t, x,**kwargs):
        # - [x] fixed downsampling factors
        # - [x] online downsampling

        timestep = t
        while timestep.dim()>1:
            timestep =  timestep[:,0]
        if timestep.dim() == 0:
            timestep = timestep.repeat(x.shape[0])
        emb = [self.all_layers[i].time_embed(timestep_embedding(timestep, self.all_layers[i].model_channels)) for i in range(3)]
        x_ = self._inference_downsample(x, min_size=self.min_size) if not isinstance(x,List) else x
        res = []
        #self._forward_nestedunet(x_, emb, res) if not self.use_prounet else self._forward_prounet(x_[-1], emb, res) # MDM NestedUnet
        if self.training_mode == 'MDM':
            self._forward_nestedunet(x_, emb, res)
        elif self.training_mode == 'MDM256x256':
            self._forward_FM256x256Unet(x_, emb, res)
        elif self.training_mode == 'progressive':
            self._forward_prounet(x_, emb, res)
        else:
            raise NotImplementedError("Only support training mode of: 'MDM', 'MDM256x256', 'progressive'")
        return res if isinstance(x, List) else res[-1]
