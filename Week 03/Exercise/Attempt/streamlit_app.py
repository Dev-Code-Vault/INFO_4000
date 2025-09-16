# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io

st.title("Covid CT Scan Classifier")

#upload image
uploaded = st.file_uploader("Upload a CT image", type=["jpg", "png", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        #send the image to Flask API
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        files = {"file": buffered}
        
        
        response = requests.post("http://localhost:5000/predict", files=files)
        if response.status_code == 200:
            preds = response.json()
            st.write(preds)
        else:
            st.error("Prediction failed: " + response.text)
