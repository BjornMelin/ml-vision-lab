import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Food Image Classification", layout="wide")

st.title("Food Image Classification")
st.write("Upload an image of food and get the predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Call the prediction API
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": uploaded_file.getvalue()},
    )

    if response.status_code == 200:
        result = response.json()
        st.write(f"Predicted class: {result['class']}")
    else:
        st.write("Error in prediction.")
