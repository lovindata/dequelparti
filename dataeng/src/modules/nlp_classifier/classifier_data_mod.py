from typing import Callable, Sequence, Tuple

import lightning as L
import torch
from torch.utils import data

from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


class ClassifierDataMod(L.LightningDataModule):
    def __init__(
        self,
        llm_rows: Sequence[LLMRowVo],
        vocabulary: VocabularyVo,
        f_prepare_feature_and_label_tensors: Callable[
            [Sequence[LLMRowVo], VocabularyVo], Tuple[torch.Tensor, torch.Tensor]
        ],
    ):
        super().__init__()
        self.llm_rows = llm_rows
        self.vocabulary = vocabulary
        self.f_prepare_feature_and_label_tensors = f_prepare_feature_and_label_tensors

    def prepare_data(self) -> None:
        feature_tensor, label_tensor = self.f_prepare_feature_and_label_tensors(
            self.llm_rows, self.vocabulary
        )
        self._tensor_dataset = data.TensorDataset(feature_tensor, label_tensor)

    def setup(self, stage: str) -> None:
        total_size = len(self._tensor_dataset)
        val_size = int(total_size * 0.15)
        test_size = int(total_size * 0.15)
        train_size = total_size - val_size - test_size
        train_dataset, val_dataset, test_dataset = data.random_split(
            self._tensor_dataset, [train_size, val_size, test_size]
        )
        self._train_loader = data.DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=11
        )
        self._val_loader = data.DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=11
        )
        self._test_loader = data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=11
        )

    def train_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
        return self._train_loader

    def val_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
        return self._val_loader

    def test_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
        return self._test_loader
