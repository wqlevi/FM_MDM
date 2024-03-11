from .unet import *
import ipdb
from typing import List
import torch

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
        input_ch = int(channel_mult[0] * model_channels)
        self.ch = input_ch
        self.input_blocks = TimestepEmbedSequential(conv_nd(dims, in_channels, self.ch, 3, padding=1))
        self.output_blocks = nn.ModuleList([])
        self.input_step_layers = nn.ModuleList([])
        self.output_step_layers = nn.ModuleList([])
        input_block_chans = []
        step_block_chans = []
        self.ds = 1

        # ----- input_blocks as downsampling ----- # yy
        #input_block_chans, self.input_blocks = self.make_up_down_blocks(self.input_blocks, channel_mult, input_block_chans)

        # ----- step blocks as downsampling ----- # yy
        #step_block_chans, self.input_step_layers = self.make_up_down_blocks(self.input_step_layers, step_channel_mult, step_block_chans, bottleneck=True)

        # ----- downsample step layers ----- #
        # middle_block fixed shape
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
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ----- upsample step layer ----- #
        #step_block_chans, self.output_step_layers = self.make_up_down_blocks(self.output_step_layers, step_channel_mult, step_block_chans, bottleneck=True, upsample=True)

        # ----- upsample output layer ----- #
        #input_block_chans, self.output_blocks = self.make_up_down_blocks(self.output_blocks, channel_mult, input_block_chans, bottleneck=False, upsample=True)

        self.out = nn.Sequential(
            normalization(self.ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

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

    def forward(self, t:Tensor, x:List[Tensor], y=None, step=0) -> List[Tensor]:
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.

        FIXME: odeint requires input and output x, k1 to be Tensor,got type: {List[Tensor]}
        """
        res = []
        hs = []

        #ipdb.set_trace()
        x_ = [x,] if not isinstance(x, List) else x
        
        assert t.ndim <=2, ValueError(f" Got unexpected t shape: {t.shape}")
        bs = x_[0].shape[0]

        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(bs)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        for h in x_:
            h = self.input_blocks(h, emb)
            hs.append(h)
        #[print(f"\033[91minput shape: {hh.shape}\033[0m") for hh in hs]
        self.forward_nested_UNet(hs, emb, None, res)
        res_out = [self.out(r) for r in res]
        return res_out if isinstance(x, List) else res_out[-1] # return the highest reso for odeint

    def f_mid(self, x, t):
        return self.middle_block(x, t)
    
    # TODO: put nn.Module inside init to allow flexible CUDA device allocation
    # TODO: skip connction needed (ie. cat(x,h))
    # TODO: add condition of num of res block before upsample
    def f_down(self, mult:List[int])-> nn.Module:
        """
        x: input tensor
        channel_mult: numbers of repetition of channel per layer
        block_channels: numbers of channel per layer
        """
        mult = 1
        layers = [
            ResBlock(self.ch,
                     self.time_embed_dim,
                     self.dropout,
                     out_channels=int(mult * self.model_channels),
                     **self.fixed_args,
                     )
        ]
        self.ch = int(mult * self.model_channels)
        layers.append(
                AttentionBlock(self.ch,
                               **self.fixed_args,
                               ))
        layers.append(Downsample(self.ch, self.conv_resample, dims=self.dims, out_channels=self.ch))
        self.ds //=2
        layer_module = TimestepEmbedSequential(*layers) # TimestepEmbedSequential is a wrapper for t and x input
        return layer_module.to("cuda:0")

    def f_up(self, mult:List[int])-> Tensor:
        """
        x: input tensor
        channel_mult: numbers of repetition of channel per layer
        block_channels: numbers of channel per layer
        """
        mult = 1
        layers = [
            ResBlock(self.ch,
                     self.time_embed_dim,
                     self.dropout,
                     out_channels=int(mult * self.model_channels),
                     **self.fixed_args,
                     )
        ]
        self.ch = int(mult * self.model_channels)
        layers.append(
                AttentionBlock(self.ch,
                               **self.fixed_args,
                               ))
        layers.append(Upsample(self.ch, self.conv_resample, dims=self.dims, out_channels=self.ch))
        self.ds //=2
        layer_module = TimestepEmbedSequential(*layers) # TimestepEmbedSequential is a wrapper for t and x input
        return layer_module.to("cuda:0")
    def f_skip(self, x:Tensor, h:Tensor)->Tensor:
        return torch.cat([x ,h], dim=1)

    def forward_nested_UNet(self,x:List[Tensor], emb:Tensor, h:Tensor=None, o=[])-> List[Tensor]:
        """
        f_up, f_down, f_mid, f_skip can be nn.Module instance
        use default channle = 32 for simplicity prototype, later changed into func arg of list
        """
        z = x[-1] if h is None else x[-1]+h

        if len(x) > 1:
            z = self.f_up(32)(self.forward_nested_UNet(x[:-1], emb, self.f_down(32)(z, emb), o), emb)
        else:
            z = self.f_up(32)(self.f_mid(self.f_down(32)(z, emb), emb), emb)
        o.append(z)
        return z

    def make_up_down_blocks(self, block_layers: nn.ModuleList,
                            channel_mult:List[int],
                            block_channels:List[int],
                            bottleneck:bool=False,
                            upsample:bool=False):
        """
        build up progressive Unet 
        """

        if not bottleneck and not upsample: block_channels.append(self.ch)
        if upsample: channel_mult = channel_mult[::-1] # flip the mult list
        res_shift=1 if upsample and not bottleneck else 0# output upsampling

        for level, mult in enumerate(channel_mult): 
            for i in range(self.num_res_blocks + res_shift):
                ich = block_channels.pop() if upsample else 0
                if upsample: print(f"Res channel {self.ch + ich}")
                layers = [
                    ResBlock(
                        self.ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        **self.fixed_args,
                    )
                ]
                self.ch = int(mult * self.model_channels)
                if self.ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            self.ch,
                            **self.fixed_args,
                        )
                    )

                if not upsample: block_channels.append(self.ch) 

                if upsample:
                    if (bottleneck and level and (i == self.num_res_blocks)) or ((not bottleneck) and (i+1 ==self.num_res_blocks)):
                        # if reached num_res_blocks
                        out_ch = self.ch
                        layers.append(
                            Upsample(self.ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                        )
                        self.ds //=2

                else:
                    if bottleneck and level == len(channel_mult)-1:
                        continue
                    else:
                        out_ch = self.ch
                        layers.append(
                            Downsample(self.ch, self.conv_resample, dims=self.dims, out_channels=out_ch))
                        self.ch = out_ch
                        self.ds *=2 
                        block_channels.append(self.ch)
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

    def forward(self, t, x, y=None, step=0, *args, **kwargs):
        return super().forward(t, x, y=y, step=step, *args)
