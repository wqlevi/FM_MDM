from .unet import *

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
        self.use_scale_shift_norm=use_scale_shift_norm,
        self.use_new_attention_order= use_new_attention_order,

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # ----- input channel for the entire UNet ----- #
        input_ch = int(channel_mult[0] * model_channels)
        self.ch = input_ch
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, self.ch, 3, padding=1))]
        )
        self._feature_size = self.ch
        step_block_chans = []
        self.ds = 1
        # input_blocks as downsampling
        """
        input_block_chans=[ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if self.ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.ds *= 2
                self._feature_size += ch
        """
        input_block_chans = self.make_up_down_blocks(input_ch, time_embed_dim)

        # ----- downsample step layers ----- #
        self.input_step_layers = nn.ModuleList([])
        for level, mult in enumerate(step_channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        self.ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                self.ch = int(mult * model_channels)
                if self.ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            self.ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_step_layers.append(TimestepEmbedSequential(*layers))
                self._feature_size += self.ch
                step_block_chans.append(self.ch)
            if not level == len(step_channel_mult) -1: # skip downsample when last mult
                out_ch = self.ch
                self.input_step_layers.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            self.ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(self.ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                self.ch = out_ch
                step_block_chans.append(self.ch)
                self.ds *= 2
                self._feature_size += self.ch



        # middle_block fixed shape
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                self.ch,
                time_embed_dim,
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
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += self.ch

        # ----- upsample step layer ----- #
        self.output_step_layer = nn.ModuleList([])
        for level, mult in list(enumerate(step_channel_mult))[::-1]:
            for i in range(num_res_blocks):
                ich = step_block_chans.pop()
                layers = [
                    ResBlock(
                        self.ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                self.ch = int(model_channels * mult)
                if self.ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            self.ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks: 
                    out_ch = self.ch
                    layers.append(
                        ResBlock(
                            self.ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(self.ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    self.ds //= 2
                self.output_step_layer.append(TimestepEmbedSequential(*layers))
                self._feature_size += self.ch

        # output_blocks as upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        self.ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                self.ch = int(model_channels * mult)
                if self.ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            self.ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                #if level and i == num_res_blocks:
                if i+1 == num_res_blocks:
                    out_ch = self.ch
                    layers.append(
                        Upsample(self.ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    self.ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += self.ch

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

    def forward(self, t, x, y=None, step=0):
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps = t
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        hs_step = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        #[print(f"\033[91minput shape: {hh.shape}\033[0m") for hh in hs]

        if step == 1:
            for module in self.input_step_layers: # step1_module
                h = module(h, emb)
                hs_step.append(h)
        #[print(f"\033[92mdown_step outputs shape: {hh.shape}\033[0m") for hh in hs_step]

        h = self.middle_block(h, emb)
        #print(f"\033[93mmid block shape: {h.shape}\033[0m")

        if step == 1:
            for module in self.output_step_layer:
                h = th.cat([h, hs_step.pop()], dim=1) 
                h = module(h, emb) 
                #print(f"\033[92mup_step block shape: {h.shape}\033[0m")

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1) # FIXME: unmatched shape
            h = module(h, emb)
            #print(f"\033[91mouput block shape: {h.shape}\033[0m")
        h = h.type(x.dtype)
        return self.out(h)

    def make_up_down_blocks(self, input_ch, time_embed_dim):
        input_block_chans = [self.ch]
        for level, mult in enumerate(self.channel_mult):
            #print(mult)
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        self.ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                self.ch = int(mult * self.model_channels)
                if self.ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            self.ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += self.ch
                input_block_chans.append(self.ch)

                out_ch = self.ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(self.ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                )
                self.ch = out_ch
                input_block_chans.append(self.ch)
                self.ds *= 2
                self._feature_size += self.ch
        return input_block_chans

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
