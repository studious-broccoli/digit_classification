import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# === Custom Functions ===
from digit_classification.models.cnn import DigitClassifier
from digit_classification.utils.model_utils import get_valid_checkpoint
from digit_classification.utils.plot_utils import plot_image
from digit_classification.utils.utils import load_config
from digit_classification.transforms import predict_transform


def predict_from_checkpoint(checkpoint_path: str = "checkpoints", input_path: str = "test.png", ) -> None:
    # === Device Setup ===
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # === Configuration Settings ===
    config = load_config()
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]
    use_cnn = config["use_cnn"]
    target_labels = config["target_labels"]

    # === Label Encoder ===
    label_encoder = LabelEncoder()
    label_encoder.fit(target_labels)

    # === Define Model ===
    model = DigitClassifier(input_dim=input_dim,
                            num_classes=num_classes,
                            use_cnn=use_cnn)
    model.to(device)

    # === Find Checkpoint ===
    resume_ckpt = get_valid_checkpoint(checkpoint_path)

    # === Load Checkpoint ===
    print(f" Loading checkpoint: {resume_ckpt}")
    model.load_state_dict(torch.load(resume_ckpt, map_location=device)["state_dict"])
    model.eval()

    # === Load and preprocess image ===
    image = Image.open(input_path)
    image_tensor = predict_transform(image).unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]
    image_tensor = image_tensor.to(device)

    # === Predict ===
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = output.argmax(dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_label])[0]

    print(f"The predicted digit is: {predicted_label}")

    out_file = input_path.replace(".png", "_prediction.png")
    title = f"Predicted: {predicted_label}"
    plot_image(image, out_file, title=title)

    return predicted_label


if __name__ == "__main__":
    predict_from_checkpoint(checkpoint_path="checkpoints", input_path="test.png")
