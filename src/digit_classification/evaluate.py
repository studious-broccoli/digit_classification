"""
    Script: evaluate.py
    Description: Evaluate the model on the test set and print out a classification report.
"""
import pdb
import os
import torch
from sklearn.preprocessing import LabelEncoder
# === Custom Functions ===
from digit_classification.data import get_dataloaders
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.utils import load_config
from digit_classification.utils.model_utils import get_valid_checkpoint
from digit_classification.utils.plot_utils import print_classification_report, plot_confusion_matrix, plot_image


def evaluate_model(data_dir: str = "data", checkpoint_path: str = "checkpoints") -> None:
    # === Device Setup ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Configuration Settings ===
    config = load_config()
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]
    target_labels = config["target_labels"]

    # === Label Encoder ===
    label_encoder = LabelEncoder()
    label_encoder.fit(target_labels)

    # === Load test_set ===
    _, _, test_loader = get_dataloaders(data_dir)

    # === Define Model ===
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)
    model.to(device)

    # === Find Checkpoint ===
    resume_ckpt = get_valid_checkpoint(checkpoint_path)

    # === Load Checkpoint ===
    print(f" Loading checkpoint: {resume_ckpt}")
    model.load_state_dict(torch.load(resume_ckpt, map_location=device)["state_dict"])
    model.eval()

    # === Data Arrays ===
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            predicted_labels = preds.argmax(dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted_labels.numpy())

    # Debugging Step (should be updated):
    # Save the first test image once as test.png
    for x_batch, y_batch in test_loader:
        first_image = x_batch[0]  # Tensor of shape [1, 28, 28]
        true_label = label_encoder.inverse_transform([y_batch[0].item()])[0]
        if true_label == 8:
            plot_image(first_image, out_file="images/test.png", title=f"Label: {true_label}")
            print("Saved one test image as images/test.png")
            break  # Only one image for testing right now

    # === Map Labels ===
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # === Classification Report ===
    print_classification_report(y_true_labels, y_pred_labels, message="... Test Classification Report ...")

    # === Confusion Matrix ===
    filename = os.path.join(checkpoint_path, "confusion_matrix_test.png")
    cm_title = plot_confusion_matrix(y_true_labels, y_pred_labels, model_name="DigitClassifier",
                                     dataset="Test", filename=filename)
    print(f"{cm_title}\nSaving Confusion Matrix: {filename}")


if __name__ == "__main__":
    evaluate_model(data_dir="data/", output_dir="checkpoints")