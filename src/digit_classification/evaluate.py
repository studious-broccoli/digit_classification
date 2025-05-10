"""
    Script: evaluate.py
    Description: Evaluate the model on the test set and print out a classification report.
"""

import os
import torch
from digit_classification.data import get_testloader
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.model_utils import get_latest_ckpt
from digit_classification.utils.plot_utils import print_classification_report, plot_confusion_matrix


def evaluate_model(data_dir="data", output_dir="checkpoints"):
    # === Load test_set ===
    test_loader = get_testloader(data_dir=data_dir)

    # === Define Model ===
    input_dim = 28 * 28
    num_classes = 10
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)

    # === Find Checkpoint ===
    checkpoint_path = get_latest_ckpt(output_dir)
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    # === Load Checkpoint ===
    print(f" Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"])
    model.eval()

    # === Data Arrays ===
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            predicted_labels = preds.argmax(dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted_labels.numpy())

    # === Classification Report ===
    print_classification_report(y_true, y_pred, message="... Test Classification Report ...")

    # === Confusion Matrix ===
    filename = os.path.join(output_dir, "Test_Confusion_Matrix.png")
    cm_title = plot_confusion_matrix(y_true, y_pred, model_name="DigitClassifier",
                                     dataset="Test", filename=filename)
    print(f"{cm_title}\nSaving Confusion Matrix: {filename}")


if __name__ == "__main__":
    evaluate_model(data_dir="data/MNIST", output_dir="checkpoints")