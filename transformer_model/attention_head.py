import torch as th


class AttentionHead(th.nn.Module):
    def __init__(self, head_size: int, block_size: int, n_embed: int, p_dropout: float):
        super().__init__()
        self.head_size = head_size
        self.key = th.nn.Linear(n_embed, head_size, bias=False)
        self.query = th.nn.Linear(n_embed, head_size, bias=False)
        self.value = th.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", th.tril(th.ones(block_size, block_size)))

        self.dropout = th.nn.Dropout(p_dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = th.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out
