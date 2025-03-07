# onmt/decoders/gtransformer_decoder.py
from onmt.decoders.transformer import TransformerDecoder, TransformerDecoderBase # ADDED import
from onmt.decoders import register_decoder # ADDED import
from onmt.modules.position_ffn import ActivationFunction # ADDED import

@register_decoder(name='gtransformer') # ADDED
class GTransformerDecoder(TransformerDecoder): # ADDED inherit from TransformerDecoder
    """GTransformer decoder inherit from TransformerDecoder."""

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        relative_positions_buckets,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ):
        super(GTransformerDecoder, self).__init__( # ADDED inherit from TransformerDecoder
            num_layers,
            d_model,
            heads,
            d_ff,
            copy_attn,
            self_attn_type,
            dropout,
            attention_dropout,
            embeddings,
            max_relative_positions,
            relative_positions_buckets,
            aan_useffn,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            add_ffnbias=add_ffnbias,
            parallel_residual=parallel_residual,
            shared_layer_norm=shared_layer_norm,
            layer_norm=layer_norm,
            norm_eps=norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
            sliding_window=sliding_window,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        self.transformer_layers = nn.ModuleList(
            [
                GTransformerDecoderLayer( # CHANGED TransformerDecoderLayer -> GTransformerDecoderLayer
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    num_kv=num_kv,
                    add_ffnbias=add_ffnbias,
                    parallel_residual=parallel_residual,
                    shared_layer_norm=shared_layer_norm,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    parallel_gpu=parallel_gpu,
                    sliding_window=sliding_window,
                    rotary_interleave=rotary_interleave,
                    rotary_theta=rotary_theta,
                    rotary_dim=rotary_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for i in range(num_layers)
            ]
        )

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.relative_positions_buckets,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias,
            num_kv=opt.num_kv,
            add_ffnbias=opt.add_ffnbias,
            parallel_residual=opt.parallel_residual,
            shared_layer_norm=opt.shared_layer_norm,
            layer_norm=opt.layer_norm,
            norm_eps=opt.norm_eps,
            use_ckpting=opt.use_ckpting,
            parallel_gpu=opt.world_size
            if opt.parallel_mode == "tensor_parallel"
            else 1,
            sliding_window=opt.sliding_window,
            rotary_interleave=opt.rotary_interleave,
            rotary_theta=opt.rotary_theta,
            rotary_dim=opt.rotary_dim,
            num_experts=opt.num_experts,
            num_experts_per_tok=opt.num_experts_per_tok,
            )

    def forward(self, tgt, enc_out=None, step=None, **kwargs):
        """
        Decode, possibly stepwise.
        when training step is always None, when decoding, step increases
        tgt (Tensor): batch x tlen x feats
        enc_out (Tensor): encoder output (batch x slen x model_dim)
        """
        if enc_out is None:
            enc_out = self.embeddings(tgt)
        if step == 0:
            self._init_cache(enc_out)
        elif step is None:
            for layer in self.transformer_layers:
                if isinstance(layer.self_attn, AverageAttention):
                    layer.self_attn.layer_cache = False, {"prev_g": torch.tensor([])}
                else:
                    layer.self_attn.layer_cache = (
                        False,
                        {"keys": torch.tensor([]), "values": torch.tensor([])},
                    )
                layer.context_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )

        dec_out = self.embeddings(tgt, step=step)

        pad_idx = self.embeddings.word_padding_idx
        src_len = kwargs["src_len"]
        src_max_len = self.state["src"].shape[1]
        src_pad_mask = sequence_mask(src_len, src_max_len).unsqueeze(
            1
        )  # [B x 1 x slen]
        tgt_pad_mask = tgt[:, :, 0].eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or self._copy or kwargs.pop("return_attn", False)

        attn_aligns = []

        # ADDED group tags from kwargs
        tgt_tags = kwargs.get("tgt_tags", None)
        src_tags = kwargs.get("src_tags", None)

        for layer in self.transformer_layers:
            dec_out, attn, attn_align = layer(
                dec_out,
                enc_out,
                src_pad_mask,
                tgt_pad_mask,
                step=step,
                future=False, # ADDED
                return_attn=return_attn,
                local_attn_mask=tgt_tags, # ADDED local_attn_mask
                global_attn_mask=tgt_tags, # ADDED global_attn_mask
                encoder_local_mask=src_tags, # ADDED encoder_local_mask
                with_align=with_align,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        dec_out = self.layer_norm(dec_out)

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_out, attns