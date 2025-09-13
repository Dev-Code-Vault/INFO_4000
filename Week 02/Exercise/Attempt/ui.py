# ui.py

#import necessary libraries
import streamlit as st
import requests

st.title("NFL Winning Season Predictor")

#input fields
pf = st.number_input("Points For (PF)", min_value=0, value=400)
pa = st.number_input("Points Against (PA)", min_value=0, value=350)
pd = st.number_input("Points Differential (PD)", min_value=-200, value=50)
sos = st.number_input("Strength of Schedule (SoS)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

team_name = st.text_input("Team Name", value="Atlanta Vipers")

#predict
if st.button("Predict"):
    input_data = {
        'features': [pf, pa, pd, sos],
        'team': team_name
    }

    #sending data to API
    try:
        response = requests.post("http://localhost:5000/predict", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        pred = prediction_result['prediction'][0]

        if pred == 1:
            st.success(f"{team_name} is predicted to have a **Winning Season**")
        else:
            st.error(f"{team_name} is predicted to have a **Losing Season**")
    except requests.exceptions.ConnectionError:
        st.error("Flask API not running. Start `python app.py` first.")
