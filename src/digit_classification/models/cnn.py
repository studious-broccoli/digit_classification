import pdb
import numpy as np
from lightning.pytorch import LightningModule
from torchmetrics.classification import Accuracy, Precision, Recall
from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn


class DigitClassifier(LightningModule):
    def __init__(
        self,
        input_dim: Optional[int] = None,  # used if using MLP
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        use_cnn: bool = True,
        class_weights: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.use_cnn = use_cnn

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None
        )

        if use_cnn:
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        else:
            assert input_dim is not None, "input_dim must be set when use_cnn=False"
            self.model = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        if self.use_cnn and x.dim() == 3:
            x = x.unsqueeze(1)
        elif not self.use_cnn:
            x = x.view(x.size(0), -1)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        weights = (
            self.class_weights.to(self.device)
            if self.class_weights is not None else None
        )
        loss = F.cross_entropy(logits, y, weight=weights)
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.val_recall(logits, y)
        self.val_precision(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_recall_macro", self.val_recall, on_epoch=True)
        self.log("val_precision_macro", self.val_precision, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)