from typing import Dict
import os

import torch as th
from omegaconf import OmegaConf

from .text_dataset_loader import TextDatasetLoader
from ..transformer_model import TransformerModel


class TransformerTrainer:
    def __init__(
        self,
        dataset_path: str,
        learning_rate: float,
        batch_size: int,
        block_size: int,
        n_eval_iterations: int,
        n_embed: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        eval_interval: int,
        train_frac: float = 0.8,
        n_tokens_to_gen_on_eval: int | None = None,
        model_save_path: str | None = None,
        seed: int | None = None,
    ):
        self._text_dataset_loader = TextDatasetLoader(
            path=dataset_path,
            batch_size=batch_size,
            block_size=block_size,
            train_frac=train_frac,
        )
        self._model = TransformerModel(
            vocab_size=self._text_dataset_loader.vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            n_heads=n_heads,
            n_layers=n_layers,
            p_dropout=p_dropout,
        )

        self._optimizer = th.optim.Adam(
            self._model.parameters(),
            lr=learning_rate
        )
        self._eval_iterations = n_eval_iterations
        self._eval_interval = eval_interval
        self._n_tokens_to_gen_on_eval = n_tokens_to_gen_on_eval
        self._save_path = model_save_path
        if self._save_path is not None:
            assert os.path.exists(os.path.dirname(self._save_path)), "Model save path does not exist"
            assert self._save_path.endswith(".pt"), "Model save path must specify a .pt file"

        if seed is not None:
            th.manual_seed(seed)

    @th.no_grad()
    def _estimate_losses(self) -> Dict[str, float]:
        losses = {}
        in_training_mode = self._model.training
        self._model.eval()
        for split, data in self._text_dataset_loader.data_splits.items():
            x, y = self._text_dataset_loader.get_batch(data)
            _, loss = self._model(x, y)
            losses[split] = loss.item()
        self._model.train(in_training_mode)
        return losses

    @th.no_grad()
    def _get_text(
        self,
        n_tokens: int,
        context: str | None = None
    ) -> str:
        if context is None:
            context = "\n"
        context = self._text_dataset_loader.encode(context)

        in_training_mode = self._model.training
        self._model.eval()
        text = self._text_dataset_loader.decode(
            self._model.generate(context, max_new_tokens=n_tokens)
        )
        self._model.train(in_training_mode)
        return text

    def train(self, n_train_iterations: int):
        for i in range(n_train_iterations):
            if i % self._eval_interval == 0:
                losses = self._estimate_losses()
                print(f"Iteration {i}: {losses}")
                print(f"Training loss: {losses['training']}, Validation loss: {losses['validation']}")

                if self._n_tokens_to_gen_on_eval is not None:
                    print("Generated text:\n")
                    print(self._get_text(n_tokens=self._n_tokens_to_gen_on_eval))
                    print("\n")

                if self._save_path is not None:
                    th.save(self._model.state_dict(), self._save_path)

            x, y = self._text_dataset_loader.get_training_batch()

            logits, loss = self._model(x, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        print("Training complete")
        if self._save_path is not None:
            th.save(self._model.state_dict(), self._save_path)

    @staticmethod
    def from_config(config: OmegaConf, load_state_dict: bool = False) -> "TransformerTrainer":
        trainer = TransformerTrainer(
            dataset_path=config.dataset.path,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            block_size=config.model.block_size,
            n_eval_iterations=config.training.n_eval_iterations,
            n_embed=config.model.n_embed,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            p_dropout=config.model.p_dropout,
            eval_interval=config.training.eval_interval,
            train_frac=config.training.train_frac,
            n_tokens_to_gen_on_eval=config.training.n_tokens_to_gen_on_eval,
            model_save_path=config.training.model_save_path,
            seed=config.training.seed,
        )

        if load_state_dict and config.training.model_save_path is not None:
            trainer._model.load_state_dict(th.load(config.training.model_save_path))

        return trainer
