from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
import torch.nn as nn


class DigitClassifier(LightningModule):
    def __init__(self, input_dim, num_classes, learning_rate=1e-3):
        super().__init__()
        # Your model implementation here
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)

        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # flatten the input
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # flatten the input
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # labels might not be needed at inference
        return self(x)