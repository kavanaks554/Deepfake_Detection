import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Deepfake Detection", page_icon="🧠")

st.title("🧠 Deepfake Detection using CNN")
st.write("Upload a face image to predict whether it is Real or Fake.")

model = load_model("deepfake_model.keras")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("Prediction: Fake")
        st.write(f"Confidence: {prediction:.4f}")
    else:
        st.success("Prediction: Real")
        st.write(f"Confidence: {1-prediction:.4f}")