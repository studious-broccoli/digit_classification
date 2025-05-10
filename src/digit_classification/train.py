# python src/digit_classification/cli.py train --data-dir data/MNIST --output-dir checkpoints --epochs 10

from datetime import datetime
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.model_utils import send_error_email, get_latest_ckpt
from digit_classification.utils.utils import load_config
from digit_classification.data import get_dataloaders


def train_model(data_dir="data", output_dir="checkpoints", epochs=20):
    # === Device Setup ===
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # === Config ===
    config = load_config()
    load_from_ckpt = config.get("load_from_ckpt", False)
    resume_training = config.get("resume_training", False)

    EMAIL_ADDRESS = config.get("EMAIL_ADDRESS", "")
    EMAIL_PASSWORD = config.get("EMAIL_PASSWORD", "")
    SMTP_SERVER = config.get("SMTP_SERVER", "")
    SMTP_PORT = config.get("SMTP_PORT", "")

    # === Load Data ===
    train_loader, val_loader, test_loader = get_dataloaders(data_dir)

    # === Model ===
    input_dim = 28 * 28
    num_classes = 10
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)

    # === Callbacks ===
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # checkpoint_callback = ModelCheckpoint(dirpath=output_dir, every_n_epochs=2, save_top_k=-1)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

    # === Checkpoint logic ===
    resume_ckpt = get_latest_ckpt(output_dir) if resume_training else None

    # === Trainer ===
    trainer = Trainer(
        max_epochs=epochs,
        default_root_dir=output_dir,
        accelerator="cpu",
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback]
    )

    # === Training ===
    try:
        if load_from_ckpt and resume_ckpt:
            print(f"[INFO] Resuming training from checkpoint: {resume_ckpt}")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
        else:
            print("[INFO] Starting training from scratch...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as e:
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        error_message = f"[ERROR] Training crashed at {timestamp}. Error: {e}"
        print(error_message)

        with open(f"./error_log_{timestamp}.txt", 'a') as f:
            f.write(error_message + '\n')

        # Optional: email notification
        # send_error_email("DigitClassifier Training Failed", error_message,
        #                  EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT)


# === Entry Point for CLI or Script Use ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # Example hardcoded test call — replace with argparse/typer if needed directly here
    train_model(data_dir="data/MNIST", output_dir="checkpoints", epochs=2)
