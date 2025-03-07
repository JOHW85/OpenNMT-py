# START_FILE: onmt/modules/g_transformer_layer.py
"""
Implementation of G-Transformer Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.group_multi_headed_attn import GroupMultiHeadedAttention #ADDED
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.rmsnorm import RMSNorm #ADDED


class GTransformerEncoderLayer(nn.Module):
    """
    GTansformer Encoder Layer Block
    """

    def __init__(self, opt):
        super(GTransformerEncoderLayer, self).__init__()
        self.enc_self_attn = GroupMultiHeadedAttention( #MODIFIED changed to GroupMultiHeadedAttention
            opt.heads,
            opt.enc_hid_size,
            dropout=opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout
        )
        self.feed_forward = PositionwiseFeedForward(opt)
        self.layer_norm = RMSNorm(opt.enc_hid_size) if opt.layer_norm == 'rms' else nn.LayerNorm(opt.enc_hid_size) #ADDED
        self.dropout = nn.Dropout(opt.dropout[0] if type(opt.dropout) is list else opt.dropout)


    def forward(self, layer_input, mask, local_attn_mask=None): #MODIFIED added local_attn_mask
        """
        Transformer Encoder Layer definition.
        Args:
            layer_input: ``torch.FloatTensor``
                * Layer input of size(seq_len, batch_size, embed_dim)
            mask: ``torch.ByteTensor``
                * mask matrix for attention.
        Returns:
            output ``torch.FloatTensor``:
                * Tensor of size=(seq_len, batch_size, embed_dim)
            output_norm ``torch.FloatTensor``:
                * Tensor of size=(seq_len, batch_size, embed_dim)
            attn ``torch.FloatTensor``:
                * Tensor of size=(batch_size, head_count, seq_len, seq_len)
        """
        input_norm = self.layer_norm(layer_input) #MODIFIED Layer Norm at the begining
        mask_norm = mask

        self_attn, attn = self.enc_self_attn( #MODIFIED added local_attn_mask
            input_norm, input_norm, input_norm,
            mask=mask_norm,
            local_attn_mask=local_attn_mask,
        )

        # Drop out and residual connect
        query = self.dropout(self_attn) + layer_input
        layer_output = self.feed_forward(query) #MODIFIED feed_forward after attention

        return layer_output, attn


    def update_dropout(self, dropout, attention_dropout):
        self.dropout.p = dropout
        self.enc_self_attn.update_dropout(attention_dropout)


class GTransformerDecoderLayer(nn.Module):
    """
    GTansformer Decoder Layer Block
    """

    def __init__(self, opt):
        super(GTransformerDecoderLayer, self).__init__()
        self.dec_self_attn = GroupMultiHeadedAttention( #MODIFIED changed to GroupMultiHeadedAttention
            opt.heads,
            opt.dec_hid_size,
            dropout=opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
        )
        self.dec_enc_attn = MultiHeadedAttention( #MODIFIED kept MultiHeadedAttention for cross-attention
            opt.heads,
            opt.dec_hid_size,
            dropout=opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
            attn_type="context",
        )
        self.feed_forward = PositionwiseFeedForward(opt)
        self.layer_norm = RMSNorm(opt.dec_hid_size) if opt.layer_norm == 'rms' else nn.LayerNorm(opt.dec_hid_size) #ADDED
        self.dropout = nn.Dropout(opt.dropout[0] if type(opt.dropout) is list else opt.dropout)
        self.context_attn = self.dec_enc_attn #MODIFIED used for alignement debug


    def forward(self, layer_input, enc_out, enc_mask, mask, step=None, layer_cache=None): #MODIFIED
        """
        Transformer Decoder Layer definition.
        Args:
            layer_input: ``torch.FloatTensor``
                * Layer input of size(seq_len, batch_size, embed_dim)
            enc_out: ``torch.FloatTensor``
                * Encoder output of size(source_len, batch_size, embed_dim)
            mask: ``torch.ByteTensor``
                * mask matrix for self-attn.
            enc_mask ``torch.LongTensor``
                * mask matrix for encoder-attn.
        Returns:
            output ``torch.FloatTensor``:
                * Tensor of size=(seq_len, batch_size, embed_dim)
            attn ``torch.FloatTensor``:
                * Tensor of size=(batch_size, head_count, seq_len, seq_len)
            align_attn ``torch.FloatTensor``:
                * Tensor of size=(batch_size, head_count, seq_len, src_len)
        """
        input_norm = self.layer_norm(layer_input) #MODIFIED Layer Norm at the begining
        mask_norm = mask
        query_norm = input_norm
        key_norm = input_norm
        value_norm = input_norm


        self_attn, attn = self.dec_self_attn( #MODIFIED
            query_norm, key_norm, value_norm,
            mask=mask_norm,
            layer_cache=layer_cache,
            step=step,
        )

        # Drop out and residual connect
        query = self.dropout(self_attn) + layer_input

        layer_output = self.feed_forward(query) #MODIFIED feed_forward after attention


        layer_output_norm = self.layer_norm(layer_output) #MODIFIED Layer Norm after attention+FFN
        context_norm = layer_output_norm
        memory_norm = enc_out
        enc_attn, align_attn = self.dec_enc_attn( #MODIFIED kept MultiHeadedAttention for cross-attention
            context_norm, memory_norm, memory_norm,
            mask=enc_mask,
            layer_cache=layer_cache,
            step=step,
        )

        # Drop out and residual connect
        layer_output = self.dropout(enc_attn) + layer_output_norm #MODIFIED residual connection with output of FFN


        return layer_output, attn #MODIFIED removed align_attn


    def update_dropout(self, dropout, attention_dropout):
        self.dropout.p = dropout
        self.dec_self_attn.update_dropout(attention_dropout)
        self.dec_enc_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)

    def init_state(self, bsz, with_cache=True): #ADDED
        """Initialize decoder layer state."""
        return self.dec_self_attn.init_state(bsz)