import io
from pathlib import Path
import sys

import streamlit as st
from PIL import Image
import torch

# Add project root to path for imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from scripts.predict import load_model, predict_image


def load_latest_model():
    """Load the latest trained model."""
    models_dir = project_root / "experiments" / "models"
    checkpoints = list(models_dir.glob("*.pth"))

    if not checkpoints:
        return None, None

    # Get latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    config_path = project_root / "configs" / "model.yaml"

    try:
        model, class_names = load_model(str(latest_checkpoint), str(config_path))
        model = model.cuda() if torch.cuda.is_available() else model
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def predict_page():
    st.title("Food Classification")
    st.write(
        "Upload a food image and the model will classify it into one of 101 "
        "categories!"
    )

    # Load model
    model, class_names = load_latest_model()

    if model is None:
        st.error("No model checkpoints found. Please train the model first!")
        return

    # File upload
    file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if file:
        try:
            # Save uploaded file temporarily
            img_path = project_root / "temp_upload.jpg"
            Image.open(io.BytesIO(file.read())).save(img_path)

            # Make prediction
            pred_class, confidence, vis_image = predict_image(
                str(img_path),
                model,
                class_names,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(file)

            with col2:
                st.subheader("Model Visualization")
                st.image(vis_image)

            # Show prediction
            st.success(
                f"Predicted Class: {pred_class} " f"(Confidence: {confidence:.2%})"
            )

            # Display top-5 predictions
            with torch.no_grad():
                # Preprocess image again
                transform = model.model.default_cfg["test_input_size"]
                input_tensor = (
                    torch.from_numpy(vis_image).permute(2, 0, 1).unsqueeze(0).float()
                    / 255.0
                )
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor, size=transform, mode="bilinear", align_corners=False
                )

                # Get predictions
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                top5_prob, top5_idx = torch.topk(probs, 5)

                # Display results
                st.subheader("Top 5 Predictions:")
                for prob, idx in zip(top5_prob[0], top5_idx[0]):
                    st.write(f"{class_names[idx]}: {prob.item():.2%}")

            # Cleanup
            img_path.unlink()

        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Instructions
    with st.expander("Usage Instructions"):
        st.markdown(
            """
        1. Upload a food image using the file uploader above
        2. The model will:
           - Predict the food category
           - Show confidence score
           - Generate a visualization highlighting important regions
           - Display top 5 probable classes
        3. For best results:
           - Use clear, well-lit images
           - Ensure food is the main subject
           - Supported formats: JPEG, PNG
        """
        )


if __name__ == "__main__":
    predict_page()
