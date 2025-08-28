# ui.py
import streamlit as st
import requests

st.title("Sklearn Model Prediction App")

X1 = st.text_input("x1 (number)")
X2 = st.text_input("x2 (number)")

if st.button("Predict"):
    try:
        input_data = {'features': [float(X1), float(X2)]}
        response = requests.post("http://localhost:5000/predict", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        st.success(f"Prediction: {prediction_result['prediction'][0]}")
    except ValueError:
        st.error("Please enter valid numbers.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Flask API. Make sure it is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during API call: {e}")
