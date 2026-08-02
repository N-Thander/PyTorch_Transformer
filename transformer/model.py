"""The full sequence-to-sequence Transformer."""

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    """Encoder-decoder Transformer, owning both stacks and the masking logic."""

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        embed_size: int = 256,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0,
        device: torch.device | str = "cpu",
        max_length: int = 100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask, shape (N, 1, 1, src_len)."""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Lower-triangular causal mask, shape (N, 1, trg_len, trg_len)."""
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Args: src (N, src_len), trg (N, trg_len). Returns (N, trg_len, trg_vocab_size)."""
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, src_mask, trg_mask)
