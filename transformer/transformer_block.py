"""The attention + feed-forward block shared by the encoder and decoder."""

import torch
import torch.nn as nn

from .self_attention import SelfAttention


class TransformerBlock(nn.Module):
    """Attention -> add & norm -> feed forward -> add & norm.

    Used directly as an encoder layer, and reused inside `DecoderBlock` as the
    cross-attention half of a decoder layer.
    """

    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
    ):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns (N, query_len, embed_size)."""
        attention = self.attention(value, key, query, mask)

        # Residual connection is on the query, so the output length follows it.
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))
