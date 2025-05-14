from sklearn.preprocessing import LabelEncoder
# === Custom Functions ===
from digit_classification.data import get_dataloaders
from digit_classification.utils.utils import load_config
from digit_classification.utils.plot_utils import plot_image


def create_test_image(data_dir: str = "data", image_path: str = "images/test.png") -> None:
    # === Configuration Settings ===
    config = load_config()
    target_labels = config["target_labels"]

    # === Label Encoder ===
    label_encoder = LabelEncoder()
    label_encoder.fit(target_labels)

    # === Load test_set ===
    _, _, test_loader = get_dataloaders(data_dir)

    # Save the first test image once as test.png
    for x_batch, y_batch in test_loader:
        first_image = x_batch[0]  # Tensor of shape [1, 28, 28]
        true_label = label_encoder.inverse_transform([y_batch[0].item()])[0]
        if true_label == 8:
            plot_image(first_image, out_file=image_path, title=None)
            print(f"Saved one test image as {image_path}")
            break  # Only one image for testing right now