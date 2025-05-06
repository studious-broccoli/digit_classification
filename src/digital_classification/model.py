import mlflow

def train(model_name, config):
    mlflow.start_run(run_name=model_name)
    model = get_model(model_name, config)
    # training code...
    mlflow.log_param("model", model_name)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()


from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

class DigitClassifier(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # labels might not be needed at inference
        return self(x)

from torchvision.models import resnet18
model = resnet18(num_classes=10)
classifier = DigitClassifier(model)

