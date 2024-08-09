from typing import Callable, Literal, Sequence, Tuple

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import adam, optimizer
from torch.utils import data

from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


class ClassifierMod(L.LightningModule):
    def __init__(
        self,
        # llm_rows: Sequence[LLMRowVo],
        # vocabulary: VocabularyVo,
        # f_prepare_feature_and_label_tensors: Callable[
        #     [Sequence[LLMRowVo], VocabularyVo], Tuple[torch.Tensor, torch.Tensor]
        # ],
    ):
        super().__init__()
        # self.llm_rows = llm_rows
        # self.vocabulary = vocabulary
        # self.f_prepare_feature_and_label_tensors = f_prepare_feature_and_label_tensors
        self._layer1 = nn.Sequential(
            nn.Linear(300, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout()
        )
        self._layer2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout()
        )
        self._layer3 = nn.Linear(128, 3)
        self._loss_fn = nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self._f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=3, average="macro"
        )

    @classmethod
    def early_stopping(cls) -> EarlyStopping:
        return EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=20,
            verbose=True,
            mode="min",
        )

    @classmethod
    def model_checkpoint(cls) -> ModelCheckpoint:
        return ModelCheckpoint(
            monitor="val_loss",
            verbose=True,
            mode="min",
        )

    def forward(self, x: torch.Tensor):
        x = self._layer1(x)
        x = self._layer2(x)
        x = self._layer3(x)
        return x

    # def prepare_data(self) -> None:
    #     feature_tensor, label_tensor = self.f_prepare_feature_and_label_tensors(
    #         self.llm_rows, self.vocabulary
    #     )
    #     self._tensor_dataset = data.TensorDataset(feature_tensor, label_tensor)

    # def setup(self, stage: str) -> None:
    #     total_size = len(self._tensor_dataset)
    #     val_size = int(total_size * 0.15)
    #     test_size = int(total_size * 0.15)
    #     train_size = total_size - val_size - test_size
    #     train_dataset, val_dataset, test_dataset = data.random_split(
    #         self._tensor_dataset, [train_size, val_size, test_size]
    #     )
    #     self._train_loader = data.DataLoader(
    #         train_dataset, batch_size=64, shuffle=True, num_workers=11
    #     )
    #     self._val_loader = data.DataLoader(
    #         val_dataset, batch_size=64, shuffle=False, num_workers=11
    #     )
    #     self._test_loader = data.DataLoader(
    #         test_dataset, batch_size=64, shuffle=False, num_workers=11
    #     )

    # def train_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
    #     return self._train_loader

    # def val_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
    #     return self._val_loader

    # def test_dataloader(self) -> data.DataLoader[Tuple[torch.Tensor, ...]]:
    #     return self._test_loader

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("test", batch)

    def configure_optimizers(
        self,
    ) -> optimizer.Optimizer:
        return adam.Adam(self.parameters(), lr=0.001)

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
        scores = {
            "loss": loss,
            "acc": acc,
            "f1": f1,
        }
        self.log_dict(
            {f"{step_kind}_{score_key}": scores[score_key] for score_key in scores},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
