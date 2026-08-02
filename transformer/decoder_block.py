"""A single decoder layer: masked self-attention followed by cross-attention."""

import torch
import torch.nn as nn

from .self_attention import SelfAttention
from .transformer_block import TransformerBlock


class DecoderBlock(nn.Module):
    """Masked self-attention over the target, then attention over encoder output."""

    def __init__(
        self,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
    ):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor | None,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x:        (N, trg_len, embed_size) target-side input
            value:    (N, src_len, embed_size) encoder output
            key:      (N, src_len, embed_size) encoder output
            src_mask: padding mask for the source sequence
            trg_mask: causal mask keeping each position from seeing the future

        Returns:
            (N, trg_len, embed_size)
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        return self.transformer_block(value, key, query, src_mask)
