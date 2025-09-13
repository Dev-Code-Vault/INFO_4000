import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Covid CT Scan Classifier")

classifier = pipeline("image-classification", model="your-username/covid-ct-scan-classifier")

uploaded = st.file_uploader("Upload a CT image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")
    if st.button("Predict"):
        preds = classifier(img)
        st.write(preds)
