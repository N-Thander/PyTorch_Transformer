"""Decoder stack: embeddings + N decoder blocks + output projection."""

import torch
import torch.nn as nn

from .decoder_block import DecoderBlock


class Decoder(nn.Module):
    """Maps target token ids plus encoder output to vocabulary logits."""

    def __init__(
        self,
        trg_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        device: torch.device | str,
        max_length: int,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x:        (N, trg_len) target token ids
            enc_out:  (N, src_len, embed_size) encoder output
            src_mask: padding mask for the source sequence
            trg_mask: causal mask for the target sequence

        Returns:
            (N, trg_len, trg_vocab_size) logits
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        return self.fc_out(x)
