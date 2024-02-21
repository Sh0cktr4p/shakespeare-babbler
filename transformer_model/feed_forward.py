import torch as th


class FeedForward(th.nn.Module):
    def __init__(self, n_embed: int, p_dropout: float):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(n_embed, n_embed * 4),
            th.nn.ReLU(),
            th.nn.Linear(n_embed * 4, n_embed),
            th.nn.Dropout(p_dropout),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)
