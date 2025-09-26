import streamlit as st
import requests
import json
import base64
import numpy as np
from PIL import Image
import io

st.title("Image Classifier")
st.write("Choose a model and image type, then upload an image for classification.")

# Create a dropdown menu for model selection
model_choice = st.selectbox(
    "Choose a model:",
    ("resnet", "vgg")
)

# Define the single URL for the Flask backend
URL = "http://127.0.0.1:5000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-Ray Image.',width='stretch')
    st.write("")
    st.write(f"Classifying an image with {model_choice}...")

    try:
        # Read the image data and convert it into bytes for sending to backend
        image_bytes = uploaded_file.getvalue()
        
        # Prepare the file and additional data for the backend
        files = {'file': image_bytes}
        data = {
            'model_name': model_choice.lower()
        }

        # Send the POST request to the single Flask API endpoint
        response = requests.post(URL, files=files, data=data)
        
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            confidence = response.json().get('confidence')

            confidence_percentage = f"{confidence:.2%}" # Convert to percentage
            
            st.success(f"The **{model_choice}** model predicts: **{prediction}**")
            st.info(f"Confidence Score: **{confidence_percentage}**")
        else:
            st.error("Error: Could not retrieve prediction or confidence from the backend.")
    except Exception as e:
        st.error(f"An error occurred: {e}")