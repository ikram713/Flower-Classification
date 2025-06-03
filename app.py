import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="🌸 Flower Classifier", layout="centered")

model = load_model("flower_model.keras")
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("🌸 Flower Classifier")
st.write("Upload an image of a flower and get its predicted class.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Updated parameter here:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)
    predicted_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100
    
    st.write("Raw model prediction vector:", prediction)
    st.write(f"### Predicted: {predicted_class} ({confidence:.2f}%)")





