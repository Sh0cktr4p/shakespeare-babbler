from .attention_head import AttentionHead
import torch as th


class MultiHeadAttention(th.nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_size: int,
        block_size: int,
        n_embed: int,
        p_dropout: float,
    ):
        super().__init__()
        self.heads = th.nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    block_size=block_size,
                    n_embed=n_embed,
                    p_dropout=p_dropout,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = th.nn.Linear(n_heads * head_size, n_embed)
        self.dropout = th.nn.Dropout(p_dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = th.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
