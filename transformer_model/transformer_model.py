import torch as th

from .transformer_block import TransformerBlock


class TransformerModel(th.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = th.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = th.nn.Embedding(block_size, n_embed)
        self.transformer_blocks = th.nn.Sequential(
            *[
                TransformerBlock(
                    n_heads=n_heads,
                    block_size=block_size,
                    n_embed=n_embed,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ],
        )
        self.ln = th.nn.LayerNorm(n_embed)
        self.lm_head = th.nn.Linear(n_embed, vocab_size)

    def forward(self, idx: th.Tensor, targets: th.Tensor | None = None) -> th.Tensor:
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(th.arange(T, device=idx.device))
        x = tok_embed + pos_embed
        x = self.transformer_blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = th.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: th.Tensor, max_new_tokens: int) -> th.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probabilities = th.nn.functional.softmax(logits, dim=-1)
            idx_next = th.multinomial(probabilities, num_samples=1)
            idx = th.cat([idx, idx_next], dim=1)
        return idx
