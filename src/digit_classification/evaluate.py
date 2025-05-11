"""
    Script: evaluate.py
    Description: Evaluate the model on the test set and print out a classification report.
"""
import pdb
import os
import torch
from digit_classification.data import get_testloader
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.model_utils import get_latest_ckpt, get_input_dim_num_classes
from digit_classification.utils.plot_utils import print_classification_report, plot_confusion_matrix, plot_image

label_map = {0: 0, 5: 1, 8: 2}
label_map_reverse = {0: 0, 1: 5, 2: 8}

def evaluate_model(data_dir="data", checkpoint_path="checkpoints"):
    # === Load test_set ===
    test_loader = get_testloader(data_dir=data_dir)

    # === Define Model ===
    input_dim, num_classes = get_input_dim_num_classes(test_loader)
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)

    # === Find Checkpoint ===
    resume_ckpt = get_latest_ckpt(checkpoint_path)
    if not resume_ckpt:
        raise FileNotFoundError(f"No checkpoint found in {resume_ckpt}")

    # === Load Checkpoint ===
    print(f" Loading checkpoint: {resume_ckpt}")
    model.load_state_dict(torch.load(resume_ckpt, map_location=torch.device('cpu'))["state_dict"])
    model.eval()

    # === Data Arrays ===
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.view(x_batch.size(0), -1)  # flatten the input
            preds = model(x_batch)
            predicted_labels = preds.argmax(dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted_labels.numpy())

    # Save the first test image once as test.png
    for x_batch, y_batch in test_loader:
        first_image = x_batch[0]  # Tensor of shape [1, 28, 28]
        true_label = label_map_reverse[y_batch[0].item()]
        plot_image(first_image, out_file="test.png", title=f"Label: {true_label}")
        print("Saved one test image as test.png")
        break  # Only do this once

    # === Map Labels ===
    y_true_labels = [label_map_reverse[p] for p in y_true]
    y_pred_labels = [label_map_reverse[p] for p in y_pred]

    # === Classification Report ===
    print_classification_report(y_true_labels, y_pred_labels, message="... Test Classification Report ...")

    # === Confusion Matrix ===
    filename = os.path.join(checkpoint_path, "Test_Confusion_Matrix.png")
    cm_title = plot_confusion_matrix(y_true_labels, y_pred_labels, model_name="DigitClassifier",
                                     dataset="Test", filename=filename)
    print(f"{cm_title}\nSaving Confusion Matrix: {filename}")


if __name__ == "__main__":
    evaluate_model(data_dir="data/MNIST", output_dir="checkpoints")