import os
import torch
from torchvision import transforms
from PIL import Image
from src.models.efficientnet import EfficientNetModel

def load_model(model_path: str, model_name: str, num_classes: int) -> torch.nn.Module:
    model = EfficientNetModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path: str, input_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def batch_predict(model: torch.nn.Module, image_dir: str, input_size: int) -> None:
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_tensor = preprocess_image(image_path, input_size)
        probabilities = predict(model, image_tensor)
        print(f"Predictions for {image_file}: {probabilities}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch prediction for food images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_name", type=str, default="tf_efficientnetv2_s", help="Model architecture name")
    parser.add_argument("--num_classes", type=int, default=101, help="Number of classes")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images for prediction")
    parser.add_argument("--input_size", type=int, default=224, help="Input size for the model")

    args = parser.parse_args()

    model = load_model(args.model_path, args.model_name, args.num_classes)
    batch_predict(model, args.image_dir, args.input_size)
