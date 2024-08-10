from typing import Literal, Tuple

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import adam, optimizer


class ClassifierMod(L.LightningModule):
    def __init__(self):
        super().__init__()
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
        def elastic_net_loss(
            l1_lambda: float, l2_lambda: float
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            l1_penalty = torch.sum(
                torch.stack(
                    [torch.sum(torch.abs(param)) for param in self.parameters()]
                )
            )
            l1_reg_loss = l1_lambda * l1_penalty
            l2_penalty = torch.sum(
                torch.stack([torch.sum(param**2) for param in self.parameters()])
            )
            l2_reg_loss = l2_lambda * l2_penalty
            return l1_reg_loss, l2_reg_loss

        x, y = batch
        y_pred = self.forward(x)
        loss: torch.Tensor = self._loss_fn(y_pred, y)
        l1_reg_loss, l2_reg_loss = elastic_net_loss(l1_lambda=1e-6, l2_lambda=1e-6)
        acc: torch.Tensor = self._accuracy(y_pred, y)
        f1: torch.Tensor = self._f1_score(y_pred, y)
        scores = {
            "loss": loss,
            "l1_reg_loss": l1_reg_loss,
            "l2_reg_loss": l2_reg_loss,
            "acc": acc,
            "f1": f1,
        }
        self.log_dict(
            {f"{step_kind}_{score_key}": scores[score_key] for score_key in scores},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss + l1_reg_loss + l2_reg_loss
