# onmt/encoders/g_transformer.py
# ADDED
from onmt.encoders.encoder import EncoderBase
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask
from onmt.modules.rmsnorm import RMSNorm
import torch.nn as nn
import torch


class GTransformerEncoder(EncoderBase):
    """G-Transformer encoder from paper:
    `G-Transformer for Document-level Machine Translation`
    https://github.com/baoguangsheng/g-transformer
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
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm="standard",
        norm_eps=1e-6,
    ):
        super().__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                GTransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
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
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            layer_norm=opt.layer_norm,
            norm_eps=opt.norm_eps,
        )

    def forward(self, src, src_len=None):
        """See :obj:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)
        mask = sequence_mask(src_len).unsqueeze(1)

        for layer in self.transformer:
            enc_out, _ = layer(enc_out, mask) # GTransformerEncoderLayer returns attn dict, but we are not using it here
        enc_out = self.layer_norm(enc_out)

        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class GTransformerEncoderLayer(nn.Module): # ADDED
    """
    GTransformer Encoder Layer implementation in ONMT.
    """
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm="standard",
        norm_eps=1e-6,
    ):
        super(GTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention( # Global Multiheaded Attention
            heads,
            d_model,
            dropout=attention_dropout,
            attn_type="self",
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
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

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
        context, attn = self.self_attn(
            norm_layer_in, norm_layer_in, norm_layer_in, mask=mask
        )
        layer_out = self.feed_forward(self.dropout(context) + layer_in)
        return layer_out, attn

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout