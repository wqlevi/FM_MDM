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
        self.net_mid =  UNetModel(dim=(3,128,128), num_channels=128, attention_resolutions="2", channel_mult=[1,2] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid=True).to(rank) # config at I2SB
        #self.net_outer =  UNetModel(dim=(3,256,256), num_channels=64, attention_resolutions="2", channel_mult=[1,2,4] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid = True).to("cuda:0") # config at MDM-S64S256
        self.net_outer =  UNetModel(dim=(3,256,256), num_channels=64, attention_resolutions="2", channel_mult=[1,2] , num_heads=8, num_head_channels=64, num_res_blocks=2, skip_mid = True).to(rank) # config at MDM-S64S128S256
        self.all_layers = [self.net_inner, self.net_mid, self.net_outer]
        self.dtype = torch.float32
        self.min_size = 32 if not 'min_size' in kwargs else kwargs['min_size']

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

    def _forward_nestedunet(self, z:List[Tensor], emb:List[Tensor], o:List[Tensor]=[]):
        if  len(z)==1:
            h_64 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(1)][0]
            hs_64 = [h_64]
            h_64 = self.f_up(self.f_mid(self.f_down(h_64, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            h_64 = self.all_layers[0].out(h_64)
            o.append(h_64)
        elif len(z)==2:
            h_64, h_128 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(2)]
            hs_128, hs_64 = [h_128], [h_64]
            h_128 = self.f_down(h_128, emb[1], self.all_layers[1].input_blocks, hs_128)
            h_64 = self.f_up(self.f_mid(self.f_down(h_64+h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            h_64 = self.all_layers[0].out(h_64)
            o.append(h_64) # final output img
            h_128 = self.f_up(h_128, emb[1], self.all_layers[1].output_blocks, hs_128)
            h_128 = self.all_layers[1].out(h_128)
            o.append(h_128)
        elif len(z)==3:
            h_64, h_128, h_256 = [self.all_layers[i].in_layer(z[i], emb[i]) for i in range(3)]
            hs_256, hs_128, hs_64 = [h_256], [h_128], [h_64]
            h_256 = self.f_down(h_256, emb[2], self.all_layers[2].input_blocks, hs_256)
            h_128 = self.f_down(h_128+h_256, emb[1], self.all_layers[1].input_blocks, hs_128)
            h_64 = self.f_up(self.f_mid(self.f_down(h_64+h_128, emb[0], self.all_layers[0].input_blocks, hs_64), emb[0]),
                          emb[0],
                          self.all_layers[0].output_blocks,
                          hs_64)
            h_64 = self.all_layers[0].out(h_64)
            o.append(h_64)
            h_128 = self.f_up(h_128, emb[1], self.all_layers[1].output_blocks, hs_128)
            h_128 = self.all_layers[1].out(h_128)
            o.append(h_128)
            h_256 = self.f_up(h_256, emb[2], self.all_layers[2].output_blocks, hs_256)
            h_256 = self.all_layers[2].out(h_256)
            o.append(h_256)
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
        #emb_64 = self.net_inner.time_embed(timestep_embedding(timestep, self.net_inner.model_channels))
        #emb_128 = self.net_mid.time_embed(timestep_embedding(timestep, self.net_outer.model_channels))
        #emb_256 = self.net_outer.time_embed(timestep_embedding(timestep, self.net_outer.model_channels))
        x_ = self._inference_downsample(x, min_size=self.min_size) if not isinstance(x,List) else x
        res = []
        self._forward_nestedunet(x_, emb, res)
        return res if isinstance(x, List) else res[-1]
