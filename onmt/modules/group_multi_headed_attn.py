# START_FILE: onmt/modules/group_multi_headed_attn.py
""" Group Multi-Head Attention module """
import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.functional import log_softmax, scaled_dot_product_attention, softmax
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import skip_init
from onmt.modules.alibi_position_bias import AlibiPositionalBias
from torch.distributed import all_reduce
from importlib import import_module

from onmt.modules.multi_headed_attn import MultiHeadedAttention #ADDED


class GroupMultiHeadedAttention(MultiHeadedAttention): #ADDED
    """Group Multi-Head Attention module from "G-Transformer for Document-level Machine Translation"
    :cite:`DBLP:journals/corr/abs-2108-09842`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward( #MODIFIED added g_key_bias, g_query_bias
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        mask: Optional[Tensor] = None,
        local_attn_mask: Optional[Tensor] = None, #ADDED
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           local_attn_mask: local mask excluding tokens with different group tags #ADDED
           step (int): decoding step (used for Rotary embedding)
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # 1) Project key, value, and query.
        batch_size = key.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # linear transformation
        query = self.linear_query(query)
        key = self.linear_keys(key)
        value = self.linear_values(value)

        # reshape to batch x num_heads x seq_len x dim_per_head
        query = query.view(batch_size, query_len, head_count, dim_per_head).transpose(1, 2)
        key = key.view(batch_size, key_len, head_count, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, key_len, head_count, dim_per_head).transpose(1, 2)

        query /= sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        scores = torch.matmul(query, key.transpose(2, 3))


        if local_attn_mask is not None: #ADDED
            scores = scores.masked_fill(local_attn_mask.unsqueeze(1).expand_as(scores), float("-inf")) #ADDED

        scores = scores.float()

        if mask is not None:
            mask = mask.expand(-1, self.head_count, -1, -1)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype) # [bs, n_heads, q_len, k_len]
        drop_attn = self.dropout(attn)

        attn_output = torch.matmul(drop_attn, value)

        context = unshape(attn_output) # [bs, q_len, n_heads * dim_per_head = model_dim]
        attn_output = self.final_linear(context) # [bs, q_len, model_dim]

        if self.parallel_gpu > 1:
            all_reduce(attn_output)

        return attn_output, attn