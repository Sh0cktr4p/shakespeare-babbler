from typing import Dict, List

import torch as th


class TextDatasetLoader:
    def __init__(
        self,
        path: str,
        batch_size: int,
        block_size: int,
        train_frac: float = 0.8,
        device: str = "cpu",
    ):
        self._path = path
        self._train_frac = train_frac
        self._batch_size = batch_size
        self._block_size = block_size
        self._dataset_text = self._read_dataset_text(path)
        self._chars = self._get_char_list(self._dataset_text)
        self._stoi = {ch: i for i, ch in enumerate(self._chars)}
        self._itos = {i: ch for i, ch in enumerate(self._chars)}
        self._data = self.encode(self._dataset_text)
        self._device = device

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def training_data(self) -> th.Tensor:
        return self._data[: int(len(self._data) * self._train_frac)]

    @property
    def validation_data(self) -> th.Tensor:
        return self._data[int(len(self._data) * self._train_frac):]

    @property
    def data_splits(self) -> Dict[str, th.Tensor]:
        return {
            "training": self.training_data,
            "validation": self.validation_data,
        }

    def encode(self, text: str) -> th.Tensor:
        return th.tensor([self._stoi[ch] for ch in text], dtype=th.long).to(self._device)

    def decode(self, indices: th.Tensor) -> str:
        return "".join([self._itos[i] for i in indices.cpu().numpy()])

    def get_batch(self, data: th.tensor):
        ix = th.randint(len(data) - self._block_size, (self._batch_size,))
        x = th.stack([data[i: i + self._block_size] for i in ix])
        y = th.stack([data[i + 1: i + self._block_size + 1] for i in ix])
        return x, y

    def get_training_batch(self) -> th.Tensor:
        return self.get_batch(self.training_data)

    def get_validation_batch(self) -> th.Tensor:
        return self.get_batch(self.validation_data)

    @staticmethod
    def _read_dataset_text(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    @staticmethod
    def _get_char_list(self, text: str) -> List[str]:
        return sorted(list(set(text)))
