from datetime import datetime
from typing import Optional
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
# === Custom Functions ===
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.model_utils import send_error_email, get_latest_ckpt
from digit_classification.utils.plot_utils import plot_learning_curves
from digit_classification.utils.utils import load_config
from digit_classification.data import get_dataloaders


def train_model(data_dir: str = "data", output_dir: str = "checkpoints", epochs: int = 20) -> None:
    # === Device Setup ===
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # === Config ===
    config = load_config()
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]
    load_from_ckpt = config.get("load_from_ckpt", False)
    resume_training = config.get("resume_training", False)
    max_epochs = config["max_epochs"]

    # === Load Data ===
    train_loader, val_loader = get_dataloaders(data_dir)

    # === Model ===
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)

    # === Callbacks ===
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

    # === Checkpoint logic ===
    resume_ckpt = get_latest_ckpt(output_dir) if resume_training else None

    # === Trainer ===
    trainer = Trainer(
        max_epochs=min(max_epochs, epochs),
        default_root_dir=output_dir,
        accelerator=device,
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

        # Plot learning curves
        log_dir = getattr(trainer.logger, "log_dir", None)
        if log_dir:
            plot_learning_curves(log_dir)
        else:
            print("[WARNING] Logger has no log_dir — skipping learning curve plot.")

    except Exception as e:
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        error_message = f"[ERROR] Training crashed at {timestamp}. Error: {e}"
        print(error_message)

        with open(f"./error_log_{timestamp}.txt", 'a') as f:
            f.write(error_message + '\n')

        # === Email notification ===
        # EMAIL_ADDRESS = config.get("EMAIL_ADDRESS", "")
        # EMAIL_PASSWORD = config.get("EMAIL_PASSWORD", "")
        # SMTP_SERVER = config.get("SMTP_SERVER", "")
        # SMTP_PORT = config.get("SMTP_PORT", "")
        # send_error_email("DigitClassifier Training Failed", error_message,
        #                  EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT)


# === Entry Point for CLI or Script Use ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # Example hardcoded test call — replace with argparse/typer if needed directly here
    train_model(data_dir="data/MNIST", output_dir="checkpoints", epochs=2)
