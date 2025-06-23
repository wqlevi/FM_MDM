from .unet import *
import ipdb
from typing import List, Tuple
import torch
import numpy as np
import wandb
import copy

from torchvision.transforms import Resize

def print_shape_wrapper(layer_name:str=""):
    def wrapper_fn(fn):
        def inner_fn(self, *args):
            [print(f"\033[1m{layer_name}\033\n[0m\033[93mtensor input shape: {arg.shape}\033[0m") if isinstance(arg, torch.Tensor) else print(f"\033[91m---\n{layer_name} Layer shape: {arg}\n---\033[0m") for arg in args]
            out = fn(self, *args)
            print(f"\033[93mtensor output shape: {out.shape}\033[0m")
            return out
        return inner_fn
    return wrapper_fn

class ProUNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        step_channel_mult=(2,4,6), # the incremental feature map by step 
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.dims = dims
        self.channel_mult = channel_mult
        self.step_channel_mult = step_channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm=use_scale_shift_norm
        self.use_new_attention_order= use_new_attention_order

        self.fixed_args = {'dims': self.dims,
                           'use_checkpoint': self.use_checkpoint,
                           'use_scale_shift_norm': self.use_scale_shift_norm,
                           'use_new_attention_order': self.use_new_attention_order,
                           'num_heads': self.num_heads,
                           'num_head_channels': self.num_head_channels,
                           }
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.time_embed_dim)
        
        # ----- input channel for the entire UNet ----- #
        input_ch = int(channel_mult[-1] * model_channels)
        self.ch = input_ch
        self.in_block= TimestepEmbedSequential(conv_nd(dims, in_channels, self.ch, 3, padding=1))
        self.input_blocks = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])
        input_block_chans = []
        self.ds = 1

        # ----- input_blocks as downsampling ----- # 
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                self.ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                self.ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                self.ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                out_channels=2*self.ch,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        # ----- input_blocks as downsampling ----- # 
        input_block_chans, self.input_blocks = self.make_up_down_blocks(self.input_blocks, channel_mult, input_block_chans)
        # ----- upsample output layer ----- #
        input_block_chans, self.output_blocks = self.make_up_down_blocks(self.output_blocks, channel_mult, input_block_chans, bottleneck=False, upsample=True)

        n = 2 # num of res blocks per upsample
        self.output_blocks = nn.ModuleList([TimestepEmbedSequential(*self.output_blocks[i:i+n]) for i in range(0,len(self.output_blocks), n)])

        self.out = nn.Sequential(
            normalization(self.ch*2),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch*2, out_channels, 3, padding=1)),
        )
        self.output_blocks = self.output_blocks[::-1]
        #self.input_blocks = self.input_blocks[::-1]

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def _inference_downsample(self, x:Tensor, min_size:int=8)->List[Tensor]:
        ls_down = [Resize(2**(s+3))(x) for s in range(int(np.log2(x.shape[-1]//min_size)))]
        #return [l for sb in ls for l in sb] # -[x] DUMB design: Tensor is a List obj it self!
        return [*ls_down, x]

    # -[x]: hook func prints layers used at inference
    # -[x]: the integration only takes one resolution, which causes 16x16 to use 8x8 layers; change it to use a list of downsampling of the largest resolution
    # -[x]: generate list of downsampling images here at inference time
    def forward(self, t:Tensor, x:List[Tensor],*args, **kwargs ) -> List[Tensor]:
        """Apply the model to an input batch.

        :param x: an [N, C, ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N, C, ...] Tensor of outputs.

        FIXME: odeint requires input and output x, k1 to be Tensor,got type: {List[Tensor]}
        """
        res = []
        hs = []

        x_ = self._inference_downsample(x) if not isinstance(x, List) else x
        
        assert t.ndim <=2, ValueError(f" Got unexpected t shape: {t.shape}")
        bs = x_[0].shape[0]

        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(bs)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        for h in x_:
            h = self.in_block(h, emb)
            hs.append(h)
        
        self.nestUNet_debug(hs, emb, res)
        #self.nestUNet(hs, emb, self.output_blocks, self.input_blocks,  None, res)
        res_out = [self.out(r) for r in res]
        return res_out if isinstance(x, List) else res_out[-1] # return the highest reso for odeint

    #@print_shape_wrapper("mid")
    def f_mid(self, x, t):
        return self.middle_block(x, t)

    @torch.no_grad
    def hook_fn(self, m:nn.Module, i:Tuple[Tensor, Tensor] ,o:Tensor):
        block_res = "layer_res_" + str(i[0].shape[-1]) + "_"
        def _get_weight_mean(m):
            for n,m in m[0].named_parameters():
                if "weight" in n:
                    print(f"\033[36mLayer {block_res+n}:{m.mean():.3f}\033[0m")
                    #wandb.log({block_res+n:m.mean()})
        print(f"\033[93mMean of input tensors: {i[0].mean():.3f}\tin shape of: {i[0].shape}\033[0m")
        print(f"\033[91mMean of output tensors: {o.mean():.3f}\tin shape of: {o.shape}\033[0m")
        #_get_weight_mean(m)
    
    #@print_shape_wrapper("down")
    def f_down(self,x, t, layer)-> Tensor:
        #print(f'\033[1mDOWN\33[0m')
        #print(f'\033[1m{id(layer)}\33[0m')
        #layer.register_forward_hook(self.hook_fn)
        return layer(x,t)

    #@print_shape_wrapper("up")
    def f_up(self,x, t, layer)-> Tensor:
        #print(f'\033[1mUP\33[0m')
        #print(f'\033[1m{id(layer)}\33[0m')
        #layer.register_forward_hook(self.hook_fn)
        return layer(x,t)

    def f_skip(self, x:Tensor, h:Tensor)->Tensor:
        return torch.cat([x ,h], dim=1)

    """
    def nestUNet_normal(self, z:List[Tensor], emb:Tensor, layers_up:nn.Module, layers_down:nn.Module):
        z_ = z.copy()
        h = None
        ctr=1
        while len(z_)>1:
            x = z_.pop() if h is None else z_.pop() + h
            h = self.f_down(x,emb, layers_down.pop())
            ctr +=1 
        else:
            x = z_.pop() + h
            h = self.f_mid(self.f_down(x,emb, layers_down.pop()), emb)
            for i in range(ctr):
                h = self.f_up(x,emb, layers_up.pop())
        return h
    """

    def nestUNet_debug_32(self, z, emb, o=[]):
        x = z[-1] # 32x32
        h_16 = self.f_down(x, emb, self.input_blocks[-(1+2)]) # feature 16x16
        h_8 = self.f_down(h_16, emb, self.input_blocks[-(1+1)]) # feature 8x8
        h = self.f_skip(h_8, self.f_up(self.f_mid(self.f_down(h_8, emb, self.input_blocks[-1]),
                                                emb),
                                     emb,
                                     self.output_blocks[-1])
                        ) # output 8x8
        #o.append(h)
        h_16_o = self.f_skip(h_16, self.f_up(h, emb, self.output_blocks[-(1+1)])) # 16x16 
        x = self.f_skip(x, self.f_up(h_16_o, emb, self.output_blocks[-(1+2)])) # 32x32
        o.append(x)

    """
    def nestUNet_debug(self, z:List[Tensor], emb:Tensor, o=[]):
        # Trial 101, looks worse than trial 100 when it comes to 16x16 and 8x8
        if not len(z)>1:
            x = z[-1]
            x = self.f_skip(x, self.f_up(self.f_mid(self.f_down(x, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            )
            o.append(x)
        elif len(z) == 2:
            #FIXME low res messed up again: maybe f_skip with input image causes issue?
            x = z[-1] # 16x16
            h_8 = self.f_down(x, emb, self.input_blocks[-(1+1)])
            x = z[-(1+1)] # 8x8
            h = self.f_skip(h_8, self.f_up(self.f_mid(self.f_down(x+h_8, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            ) # output 8x8
            o.append(h)
            x = z[-1]
            x = self.f_skip(x, self.f_up(h, emb, self.output_blocks[-(1+1)])) # 16x16
            o.append(x)
        elif len(z) == 3:
            x = z[-1] # 32x32
            h_16 = self.f_down(x, emb, self.input_blocks[-(1+2)]) # 16x16 feature
            x = z[-(1+1)] # 16x16
            h_8 = self.f_down(x+h_16, emb, self.input_blocks[-(1+1)])# 8x8 feature
            x = z[-(1+2)] # 8x8
            h_8_o = self.f_skip(h_8, self.f_up(self.f_mid(self.f_down(x+h_8, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            ) # output 8x8
            o.append(h_8_o)
            x = z[-(1+1)]
            h_16_o = self.f_skip(h_16, self.f_up(h_8_o, emb, self.output_blocks[-(1+1)])) # 16x16 
            o.append(h_16_o)
            x = z[-1]
            x = self.f_skip(x, self.f_up(h_16_o, emb, self.output_blocks[-(1+2)])) # 32x32
            o.append(x)
    """

    # trial 100
    # TODO: flexible config of layers and channels
    def nestUNet_debug(self, z:List[Tensor], emb:Tensor, o=[]):
        if not len(z)>1:
            x = z[-1]
            x = self.f_skip(x, self.f_up(self.f_mid(self.f_down(x, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            )
            o.append(x)
        elif len(z) == 2:
            #FIXME low res messed up again: maybe f_skip with input image causes issue?
            x = z[-1] # 16x16
            h_8 = self.f_down(x, emb, self.input_blocks[-(1+1)])
            x = z[-(1+1)] # 8x8
            h = self.f_skip(x+h_8, self.f_up(self.f_mid(self.f_down(x+h_8, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            ) # output 8x8
            o.append(h)
            x = z[-1]
            x = self.f_skip(x, self.f_up(h, emb, self.output_blocks[-(1+1)])) # 16x16
            o.append(x)
        elif len(z) == 3:
            x = z[-1] # 32x32
            h_16 = self.f_down(x, emb, self.input_blocks[-(1+2)]) # 16x16 feature
            x = z[-(1+1)] # 16x16
            h_8 = self.f_down(x+h_16, emb, self.input_blocks[-(1+1)])# 8x8 feature
            x = z[-(1+2)] # 8x8
            h_8_o = self.f_skip(x+h_8, self.f_up(self.f_mid(self.f_down(x+h_8, emb, self.input_blocks[-1]),
                                                    emb),
                                         emb,
                                         self.output_blocks[-1])
                            ) # output 8x8
            o.append(h_8_o)
            x = z[-(1+1)]
            h_16_o = self.f_skip(x+h_16, self.f_up(h_8_o, emb, self.output_blocks[-(1+1)])) # 16x16 
            o.append(h_16_o)
            x = z[-1]
            x = self.f_skip(x, self.f_up(h_16_o, emb, self.output_blocks[-(1+2)])) # 32x32
            o.append(x)

    """
    def nestUNet(self, z:List[Tensor], emb:Tensor, layers_up:List[nn.Module], layers_down:List[nn.Module], h:Tensor=None,o:List=[]):
        # TODO: - [x] allow progressive appending image of higher reso.
        # TODO: - [x] allow progressive appending layers for higher reso.(ie. keep the smallest bottleneck reso unchanged for all image reso)
        # TODO: - [ ] higher res got more downsample calls

        x = z[-1] if h is None else z[-1]+h
        # only pass the last layer element
        #ipdb.set_trace()
        layer_up = layers_up[-1]
        layer_down = layers_down[-1]
        if len(z)>1:
            x = self.f_skip(x, self.f_up(self.nestUNet(z[:-1],
                                         emb,
                                         layers_up[:-1],
                                         layers_down[:-1],
                                         self.f_down(x, emb, layer_down),
                                         o),
                            emb,
                            layer_up))
        else:
            x = self.f_skip(x ,self.f_up(self.f_mid(self.f_down(x, emb, layer_down),
                                                    emb),
                                         emb,
                     layer_up))
        o.append(x)
        return x
    """

    def make_up_down_blocks(self, block_layers: nn.ModuleList,
                            channel_mult:List[int],
                            block_channels:List[int],
                            bottleneck:bool=False,
                            upsample:bool=False):
        """
        build up progressive Unet 
        """

        #if not upsample: block_channels.append(self.ch)
        if upsample: channel_mult = channel_mult[::-1] # flip the mult list
        res_shift=1 if upsample else 0# output to compensate mid layer

        for level, mult in enumerate(channel_mult): 
            ch = int(mult * self.model_channels)
            ich = block_channels.pop() if upsample else 0
            for i in range(self.num_res_blocks + res_shift):
                out_ch = int(mult* self.model_channels) 
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=out_ch,
                        **self.fixed_args,
                    )
                ]
                layers.append(
                        AttentionBlock(
                            ch,
                            **self.fixed_args,
                        )
                    )
                if not upsample: block_channels.append(ch) 

                if upsample:
                    if i == 0:
                    # if reached num_res_blocks
                        out_ch = ch + ich
                        layers.append(
                            Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                        )

                else:
                    #if not level == len(channel_mult)-1:
                    out_ch = ch
                    layers.append(
                        Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch))
                    ch = out_ch
                    #self.ds *=2 
                block_layers.append(TimestepEmbedSequential(*layers))
        return block_channels, block_layers

class ProUNetModelWrapper(ProUNetModel):
    def __init__(
        self,
        dim,
        num_channels,
        num_res_blocks,
        channel_mult=None,
        step_channel_mult = (2,),
        learn_sigma=False,
        class_cond=False,
        num_classes=NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    ):
        """Dim (tuple): (C, H, W)"""
        image_size = dim[-1]
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                #channel_mult = (1, 2, 2, 2)
                channel_mult = (1, 2, 2, )
            elif image_size == 28:
                #channel_mult = (1, 2, 2)
                channel_mult = (1, 2,)
            elif image_size == 16:
                channel_mult = (1, 2, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = list(channel_mult)

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return super().__init__(
            image_size=image_size,
            in_channels=dim[0],
            model_channels=num_channels,
            out_channels=(dim[0] if not learn_sigma else dim[0] * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            step_channel_mult = step_channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )

    def forward(self, t, x,*args, **kwargs):
        return super().forward(t, x, *args,  **kwargs)
