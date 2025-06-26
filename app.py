import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# Load the trained model
model = load_model("catdog_model.h5")

# Title and description
st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload a photo and the model will tell you if it's a cat or dog!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)


    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_batch)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show result
    st.markdown(f"### 🧠 Prediction: `{label}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}`")
