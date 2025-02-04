import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import yaml

from src.models.efficientnet import FoodClassifier
from src.utils.visualization import GradCAM


def load_model(
    checkpoint_path: str, config_path: str
) -> Tuple[FoodClassifier, List[str]]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config

    Returns:
        tuple: (model, class_names)
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model
    model = FoodClassifier(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load class names
    data_root = Path(config["dataset"]["root"])
    with open(data_root / "meta" / "classes.txt") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names


def predict_image(
    image_path: str, model: FoodClassifier, class_names: List[str], device: str = "cuda"
) -> Tuple[str, float, Image.Image]:
    """Predict class for a single image.

    Args:
        image_path: Path to image file
        model: Loaded model
        class_names: List of class names
        device: Device to run inference on

    Returns:
        tuple: (predicted_class, confidence, visualization)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_class = class_names[pred_idx.item()]

    # Generate visualization
    grad_cam = GradCAM(model, "model.blocks[-1]")
    heatmap = grad_cam.generate(input_tensor)
    vis_image = grad_cam.overlay_heatmap(image, heatmap)

    return pred_class, confidence.item(), vis_image


def batch_predict(
    input_dir: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
):
    """Run batch prediction on directory of images.

    Args:
        input_dir: Directory containing images
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        output_dir: Directory to save results
        device: Device to run inference on
    """
    # Load model
    print("Loading model...")
    model, class_names = load_model(checkpoint_path, config_path)
    model = model.to(device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    input_dir = Path(input_dir)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    print(f"Processing {len(image_files)} images...")
    for image_file in image_files:
        try:
            pred_class, confidence, vis_image = predict_image(
                str(image_file), model, class_names, device
            )

            # Save visualization
            output_path = output_dir / f"{image_file.stem}_pred.png"
            vis_image.save(output_path)

            print(f"{image_file.name}: {pred_class} " f"(confidence: {confidence:.2%})")

        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    args = parser.parse_args()
    batch_predict(
        args.input_dir, args.checkpoint, args.config, args.output_dir, args.device
    )
