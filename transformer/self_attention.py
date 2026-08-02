"""Multi-head self-attention."""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Scaled dot-product attention computed over `heads` parallel heads.

    The embedding is split into `heads` chunks of size `embed_size // heads`.
    Each head learns its own value/key/query projection, and the concatenated
    head outputs are mixed back to `embed_size` by a final linear layer.
    """

    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "embed_size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend `query` over `keys`/`values`.

        Args:
            values: (N, value_len, embed_size)
            keys:   (N, key_len, embed_size)
            query:  (N, query_len, embed_size)
            mask:   broadcastable to (N, heads, query_len, key_len); positions
                    holding 0 are masked out. Pass None for no masking.

        Returns:
            (N, query_len, embed_size)
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape:    (N, key_len, heads, head_dim)
        # energy shape:  (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape:    (N, value_len, heads, head_dim)
        # out shape:       (N, query_len, heads, head_dim) -> flattened

        return self.fc_out(out)
