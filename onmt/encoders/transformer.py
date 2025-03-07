"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask
from onmt.decoders.transformer import TransformerDecoderLayerBase # ADDED import

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    from onmt.modules.rmsnorm import RMSNorm


class TransformerEncoderLayer(TransformerDecoderLayerBase): # ADDED inherit from TransformerDecoderLayerBase
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        num_kv (int): number of heads for KV when different vs Q (multiquery)
        add_ffnbias (bool): whether to add bias to the FF nn.Linear
        parallel_residual (bool): Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models
        layer_norm (string): type of layer normalization standard/rms
        norm_eps (float): layer norm epsilon
        use_ckpting (List): layers for which we checkpoint for backward
        parallel_gpu (int): Number of gpu for tensor parallelism
        rotary_interleave (bool): Interleave the head dimensions when rotary
            embeddings are applied
        rotary_theta (int): rotary base theta
        rotary_dim (int): rotary dim when different to dim per head
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        max_relative_positions=0,
        relative_positions_buckets=0,
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
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerEncoderLayer, self).__init__( # ADDED inherit from TransformerDecoderLayerBase
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type="scaled-dot",
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
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
        )
        self.self_attn_local = MultiHeadedAttention( # ADDED _local
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            max_relative_positions=max_relative_positions,
            relative_positions_buckets=relative_positions_buckets,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.self_attn_global = MultiHeadedAttention( # ADDED _global
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            max_relative_positions=max_relative_positions,
            relative_positions_buckets=relative_positions_buckets,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.self_attn_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid()) # ADDED gating layer
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            add_ffnbias,
            parallel_residual,
            layer_norm,
            norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.parallel_residual = parallel_residual
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_in, mask, local_attn_mask=None, global_attn_mask=None): # ADDED local_attn_mask, global_attn_mask
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.layer_norm(layer_in)

        self_attn_local, attn_local = self._forward_self_attn( # ADDED _local
            norm_layer_in, mask, return_attn=True, local_attn_mask=local_attn_mask # ADDED local_attn_mask
        )
        self_attn_global, attn_global = self._forward_self_attn( # ADDED _global
            norm_layer_in, mask, return_attn=True, global_attn_mask=global_attn_mask # ADDED global_attn_mask
        )
        self_attn = self.self_attn_gate(torch.cat([self_attn_local, self_attn_global], dim=-1)) # ADDED gating layer
        self_attn = self_attn * self_attn_local + (1 - self_attn) * self_attn_global # ADDED gating mechanism

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)
        if self.parallel_residual:
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = (
                self.feed_forward(norm_layer_in) - norm_layer_in + layer_in + self_attn
            )
        else:
            layer_out = self_attn + layer_in
            layer_out = self.feed_forward(layer_out)

        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn_local.update_dropout(attention_dropout) # ADDED _local
        self.self_attn_global.update_dropout(attention_dropout) # ADDED _global
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

    def _forward_self_attn(self, norm_layer_in, mask, step=None, return_attn=False, local_attn_mask=None, global_attn_mask=None): # ADDED local_attn_mask, global_attn_mask
        if self.self_attn_type in ["scaled-dot", "scaled-dot-flash"]:
            return self.self_attn_local( # CHANGED self.self_attn -> self.self_attn_local
                norm_layer_in,
                norm_layer_in,
                norm_layer_in,
                mask=mask,
                attn_mask=local_attn_mask, # ADDED
                step=step,
                return_attn=return_attn,
            ), None # ADDED return None
        elif self.self_attn_type == "average":
            return self.self_attn(norm_layer_in, mask=mask, step=step), None
        else:
            raise ValueError(f"self attention {type(self.self_attn)} not supported")
Use code with caution.
Python
# onmt/models/__init__.py
__all__ = [
    "build_model_saver",
    "ModelSaver",
    "BaseModel",
    "NMTModel",
    "LanguageModel",
    "str2enc", # ADDED
    "str2dec", # ADDED
]
from onmt.encoders import str2enc # ADDED
from onmt.decoders import str2dec # ADDED
Use code with caution.
Python
# onmt/models/model_builder.py
def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    # ADDED
    if enc_type == "transformer":
        if opt.doc_mode == 'partial':
            enc_type = "gtransformer"
    return str2enc[enc_type].from_opt(opt, embeddings)

def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = (
        "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed else opt.decoder_type
    )
    # ADDED
    if dec_type == "transformer":
        if opt.doc_mode == 'partial':
            dec_type = "gtransformer"
    return str2dec[dec_type].from_opt(opt, embeddings)

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        num_kv (int): number of heads for KV when different vs Q (multiquery)
        add_ffnbias (bool): whether to add bias to the FF nn.Linear
        parallel_residual (bool): Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models
        layer_norm (string): type of layer normalization standard/rms
        norm_eps (float): layer norm epsilon
        use_ckpting (List): layers for which we checkpoint for backward
        parallel_gpu (int): Number of gpu for tensor parallelism
        rotary_interleave (bool): Interleave the head dimensions when rotary
            embeddings are applied
        rotary_theta (int): rotary base theta
        rotary_dim (int): rotary dim when different to dim per head
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        max_relative_positions=0,
        relative_positions_buckets=0,
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
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            max_relative_positions=max_relative_positions,
            relative_positions_buckets=relative_positions_buckets,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            add_ffnbias,
            parallel_residual,
            layer_norm,
            norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.parallel_residual = parallel_residual
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_in, mask):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.layer_norm(layer_in)
        context, _ = self.self_attn(
            norm_layer_in, norm_layer_in, norm_layer_in, mask=mask
        )
        if self.dropout_p > 0:
            context = self.dropout(context)
        if self.parallel_residual:
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = (
                self.feed_forward(norm_layer_in) - norm_layer_in + layer_in + context
            )
        else:
            layer_out = context + layer_in
            layer_out = self.feed_forward(layer_out)

        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        * src_len ``(batch_size)``
    """

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
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
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
                for i in range(num_layers)
            ]
        )
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

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
            rotary_interleave=opt.rotary_interleave,
            rotary_theta=opt.rotary_theta,
            rotary_dim=opt.rotary_dim,
        )

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)
        mask = sequence_mask(src_len).unsqueeze(1).unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # Padding mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.

        for layer in self.transformer:
            enc_out = layer(enc_out, mask)
        enc_out = self.layer_norm(enc_out)
        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
