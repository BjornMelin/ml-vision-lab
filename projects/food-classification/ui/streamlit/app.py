import streamlit as st
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.model import Model
from PIL import Image
import torchvision.transforms as T

def main():
    st.title("Image Classification Demo")
    
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Load model and transform
        model = Model(num_classes=10)
        model.load("experiments/models/model.pt")
        model.eval()
        
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        
        # Make prediction
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()
        
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {probabilities[0][predicted_class]:.2%}")

if __name__ == "__main__":
    main()
