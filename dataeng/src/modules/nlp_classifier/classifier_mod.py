from typing import Callable, Literal, Sequence, Tuple

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.optim import adam, optimizer
from torch.utils import data

from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


class ClassifierMod(L.LightningModule):
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
        self._fc1 = nn.Linear(300, 256)
        self._fc2 = nn.Linear(256, 128)
        self._fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.5)
        self._loss_fn = nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self._f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=3, average="macro"
        )

    def forward(self, x: torch.Tensor):
        x = F.relu(self._fc1(x))
        x = self.dropout(x)
        x = F.relu(self._fc2(x))
        x = self.dropout(x)
        x = self._fc3(x)
        return x

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("test", batch)

    def _step(
        self,
        step_kind: Literal["train", "val", "test"],
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x)
        loss: torch.Tensor = self._loss_fn(y_pred, y)
        acc: torch.Tensor = self._accuracy(y_pred, y)
        f1: torch.Tensor = self._f1_score(y_pred, y)
        self.log_dict(
            {f"{step_kind}_loss": loss, f"{step_kind}_acc": acc, f"{step_kind}_f1": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(
        self,
    ) -> optimizer.Optimizer:
        return adam.Adam(self.parameters(), lr=0.001)
