import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os

# Load the trained SVM model and scaler
model = joblib.load('svm_cats_dogs_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(128, 128)):
    image = np.array(image)
    image = cv2.resize(image, img_size)
    image = image.flatten()  # Flatten the image into a 1D vector
    image = scaler.transform([image])  # Normalize the image
    return image

# Streamlit interface
st.title('Cat vs Dog Classifier')

# Image upload option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    
    # Display the result
    label = 'Dog' if prediction[0] == 1 else 'Cat'
    st.write(f"Prediction: {label}")
