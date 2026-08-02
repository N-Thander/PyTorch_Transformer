# PyTorch — Transformer

A Transformer ("Attention Is All You Need", Vaswani et al., 2017) implemented
from scratch in PyTorch, using only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`
and `torch.einsum` — no `nn.Transformer`, no `nn.MultiheadAttention`.

Built as a learning project: every layer is written out so the data flow is
visible end to end.

## Project structure

Each class lives in its own module, and each module depends only on the layers
directly beneath it:

```
PyTorch_Transformer/
├── main.py                          # smoke test / demo entry point
├── README.md
└── transformer/
    ├── __init__.py                  # public API
    ├── self_attention.py            # SelfAttention      — multi-head attention
    ├── transformer_block.py         # TransformerBlock   — attention + feed forward
    ├── encoder.py                   # Encoder            — embeddings + N blocks
    ├── decoder_block.py             # DecoderBlock       — masked self-attn + cross-attn
    ├── decoder.py                   # Decoder            — embeddings + N blocks + fc_out
    └── model.py                     # Transformer        — encoder + decoder + masking
```

Composition, bottom up:

```
SelfAttention
    └── TransformerBlock ──────────────┐
            ├── Encoder                │
            └── DecoderBlock ──────────┤
                    └── Decoder        │
                            └── Transformer
```

`TransformerBlock` is deliberately reused in two roles: on its own it is an
encoder layer (self-attention, where value/key/query are the same tensor), and
inside `DecoderBlock` it is the cross-attention half (value/key come from the
encoder, query from the decoder).

## Hyperparameters

`Transformer` constructor arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `src_vocab_size` | — | source vocabulary size |
| `trg_vocab_size` | — | target vocabulary size |
| `src_pad_idx` | — | token id treated as padding in the source (masked out) |
| `trg_pad_idx` | — | token id treated as padding in the target |
| `embed_size` | `256` | model dimension; must be divisible by `heads` |
| `num_layers` | `6` | encoder and decoder blocks in each stack |
| `forward_expansion` | `4` | feed-forward hidden size multiplier |
| `heads` | `8` | attention heads (`head_dim = embed_size // heads`) |
| `dropout` | `0` | dropout probability |
| `device` | `"cpu"` | device used to build the position indices |
| `max_length` | `100` | maximum sequence length for positional embeddings |

Note that `device` is used internally to place the positional-index tensor, so
pass it explicitly *and* call `.to(device)` on the model.

## Implementation notes

**Masking.** Two masks are built in `Transformer`:

- `make_src_mask` — a padding mask of shape `(N, 1, 1, src_len)`, zero wherever
  the source token equals `src_pad_idx`.
- `make_trg_mask` — a lower-triangular causal mask of shape
  `(N, 1, trg_len, trg_len)` so each position can only attend to itself and
  earlier positions.

Both broadcast against the `(N, heads, query_len, key_len)` attention scores;
masked entries are filled with `-1e20` before the softmax.

**Attention with einsum.** Scores are `nqhd,nkhd->nhqk` (queries × keys) and the
weighted sum is `nhql,nlhd->nqhd` (attention × values), where
`n` = batch, `h` = heads, `q`/`k`/`l` = sequence positions, `d` = head dim.
Scores are scaled by `sqrt(head_dim)`.

**Residual connections.** In `TransformerBlock` the residual is added to the
*query*, so the output sequence length follows the query — which is what makes
the same block valid for cross-attention, where the source and target lengths
differ.


## Credits

* **Original Paper:** [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* **Reference Video:** [Aladdin Persson — PyTorch Transformer from Scratch](https://youtu.be/U0s0f995w14?si=zCp46iH6-KZhq9SQ)
