from pathlib import Path
import sys
import torch
import yaml
from PIL import Image
import torchvision.transforms as T

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.model import Model

def predict(image_path: str):
    # Load config
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load model
    model = Model(num_classes=config["model"]["num_classes"])
    model.load("experiments/models/model.pt")
    model.eval()

    # Load and preprocess image
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()

    class_id, probs = predict(args.image_path)
    print(f"Predicted class: {class_id}")
    print(f"Probabilities: {probs}")
