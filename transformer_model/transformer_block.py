import torch as th
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(th.nn.Module):
    def __init__(
        self,
        n_heads: int,
        block_size: int,
        n_embed: int,
        p_dropout: float,
    ):
        super().__init__()
        head_size = n_embed // n_heads
        self.self_attention = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            block_size=block_size,
            n_embed=n_embed,
            p_dropout=p_dropout,
        )
        self.ffwd = FeedForward(n_embed=n_embed, p_dropout=p_dropout)
        self.ln1 = th.nn.LayerNorm(n_embed)
        self.ln2 = th.nn.LayerNorm(n_embed)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
