# onmt/decoders/g_transformer.py
# ADDED
import torch.nn as nn
import torch.nn.functional as F

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward, ActivationFunction
from onmt.utils.misc import sequence_mask
from onmt.modules.rmsnorm import RMSNorm


class GTransformerDecoder(DecoderBase): # ADDED
    """
    GTransformer Decoder implementation in ONMT.
    """
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
        alignment_layer,
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm="standard",
        norm_eps=1e-6,
    ):
        super(GTransformerDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer, layer_norm, norm_eps
        )

        self.transformer_layers = nn.ModuleList(
            [
                GTransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )

    @classmethod
    def from_opt(cls, opt, embeddings):
        """ See :func:`TransformerDecoderBase.from_opt()`"""
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
            opt.alignment_layer,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            layer_norm=opt.layer_norm,
            norm_eps=opt.norm_eps,
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, enc_out=None, step=None, **kwargs):
        """ See :obj:`DecoderBase.forward()`"""
        if step == 0:
            self._init_cache(enc_out)

        dec_out = self.embeddings(tgt, step=step)

        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt[:, :, 0].eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or self._copy or kwargs.pop("return_attn", False)

        attn_aligns = []

        for layer in self.transformer_layers:
            dec_out, attn, attn_align = layer(
                dec_out,
                enc_out,
                tgt_pad_mask,
                step=step,
                with_align=with_align,
                return_attn=return_attn,
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


class GTransformerDecoderLayer(nn.Module): # ADDED
    """
    GTransformer Decoder Layer implementation in ONMT.
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm="standard",
        norm_eps=1e-6,
    ):
        super(GTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention( # Global Multiheaded Attention
            heads,
            d_model,
            dropout=attention_dropout,
            attn_type="self",
            self_attn_type=self_attn_type,
        )

        self.context_attn = MultiHeadedAttention( # Global Multiheaded Attention
            heads,
            d_model,
            dropout=attention_dropout,
            attn_type="context",
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            activation_fn=pos_ffn_activation_fn,
            layer_norm=layer_norm,
            norm_eps=norm_eps,
        )
        if layer_norm == "standard":
            self.layer_norm_1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm_1 = RMSNorm(d_model, eps=norm_eps)
            self.layer_norm_2 = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout


    def forward(
        self,
        layer_in,
        enc_out,
        tgt_pad_mask,
        step=None,
        future=False,
        return_attn=False,
    ):
        """Transformer Decoder layer block."""
        dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
        src_pad_mask = None # unused

        norm_layer_in = self.layer_norm_1(layer_in)
        query = self.dropout(self.self_attn(
            norm_layer_in, norm_layer_in, norm_layer_in,
            mask=dec_mask)[0]) + layer_in

        norm_query = self.layer_norm_2(query)
        layer_out, attn = self.context_attn(
            enc_out, enc_out, norm_query,
            mask=src_pad_mask,
            return_attn=return_attn,
        )
        layer_out = self.feed_forward(self.dropout(layer_out) + query)

        return layer_out, attn, None

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:
            # Add triangular future_mask and pad_mask
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.tril_(0).view(1, tgt_len, tgt_len)
            future_mask = future_mask.type_as(tgt_pad_mask)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:
            # Only mask padding.
            dec_mask = tgt_pad_mask
        return dec_mask

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout