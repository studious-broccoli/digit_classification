import torch
from PIL import Image
# === Custom Functions ===
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.plot_utils import plot_image
from digit_classification.utils.utils import load_config
from digit_classification.transforms import predict_transform


def predict_from_checkpoint(checkpoint_path="checkpoints", input_path="test.png"):
    config = load_config()
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]

    # === Define Model ===
    model = DigitClassifier(input_dim=input_dim, num_classes=num_classes)

    # === Load Checkpoint ===
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"])
    model.eval()

    # === Load and preprocess image ===
    image = Image.open(input_path)
    image_tensor = predict_transform(image).unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]
    image_flat = image_tensor.view(image_tensor.size(0), -1)  # flatten the input

    # === Predict ===
    with torch.no_grad():
        output = model(image_flat)
        predicted_label = output.argmax(dim=1).item()

    print(f"The predicted digit is: {predicted_label}")

    out_file = input_path.replace(".png", "_prediction.png")
    title = f"Predicted: {predicted_label}"
    plot_image(image, out_file, title=title)

    return predicted_label


if __name__ == "__main__":
    predict_from_checkpoint(checkpoint_path="checkpoints", input_path="test.png")
