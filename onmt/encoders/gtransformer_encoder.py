# onmt/encoders/gtransformer_encoder.py
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders import register_encoder

@register_encoder(name='gtransformer') # ADDED
class GTransformerEncoder(TransformerEncoder): # ADDED inherit from TransformerEncoder
    """GTransformer encoder inherit from TransformerEncoder."""

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.relative_positions_buckets,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias,
            num_kv=opt.num_kv,
            add_ffnbias=opt.add_ffnbias,
            parallel_residual=opt.parallel_residual,
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
        )

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        relative_positions_buckets,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
    ):
        super(GTransformerEncoder, self).__init__( # ADDED inherit from TransformerEncoder
            num_layers,
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            embeddings,
            max_relative_positions,
            relative_positions_buckets,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            add_ffnbias=add_ffnbias,
            parallel_residual=parallel_residual,
            layer_norm=layer_norm,
            norm_eps=norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
        )

    def forward(self, src, src_len=None, src_tag=None): # ADDED src_tag
        """See :func:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)
        mask = sequence_mask(src_len).unsqueeze(1).unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # Padding mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.

        local_attn_mask = None # ADDED
        global_attn_mask = None # ADDED

        if src_tag is not None: # ADDED
            local_attn_mask = src_tag.unsqueeze(1) != src_tag.unsqueeze(2) # ADDED
            local_attn_mask &= 0 != src_tag.unsqueeze(2) # ADDED

        for layer in self.transformer:
            enc_out = layer(enc_out, mask, local_attn_mask=local_attn_mask, global_attn_mask=global_attn_mask) # MODIFIED pass local_attn_mask, global_attn_mask
        enc_out = self.layer_norm(enc_out)
        return enc_out, None, src_len