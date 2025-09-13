import sqlite3
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

DB_PATH = Path('NFL.db')       
MODEL_PATH = Path('model.pkl') # produced by the notebook
TABLE = 'stats'
STAT_COLS = ['PF','PA','PD','SoS']

st.set_page_config(page_title="NFL Season Outcome Predictor", page_icon="🏈", layout="centered")
st.title("🏈 NFL Season Outcome Predictor")
st.caption("Enter team stats to predict a winning (1) vs non-winning (0) season. Inputs and predictions are stored in the database.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# DB helpers
def get_conn():
    return sqlite3.connect(DB_PATH)

# Main app
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}. Run the notebook first. Error: {e}")
    st.stop()

with get_conn() as conn:

    st.subheader("Add / Predict a Team")
    with st.form("predict_form", clear_on_submit=False):
        tm = st.text_input("Team name:")
        pf = st.number_input("PF (Points For)", min_value=0, step=1, value=465)
        pa = st.number_input("PA (Points Against)", min_value=0, step=1, value=380)
        pdiff = st.number_input("PD (Point Differential)", step=1.0, value=85.0, help="Usually PF - PA")
        sos = st.number_input("SoS (Strength of Schedule)", step=0.1, value=1.5)
        submitted = st.form_submit_button("Predict & Save")
    
    if submitted:
        try:
            input_df = pd.DataFrame({'PF':[pf],'PA':[pa],'PD':[pdiff],'SoS':[sos]})
            pred = int(model.predict(input_df)[0])
            st.success(f"Prediction for **{tm}**: **{pred}** (1=Winning season, 0=Non-winning)")
            
            row = pd.DataFrame({
                'Tm':[tm],
                'W':[None],
                'L':[None],
                'PF':[pf],
                'PA':[pa],
                'PD':[pdiff],
                'SoS':[sos],
                'Winning_Season':[pred]
            })
            
            row.to_sql(TABLE, conn, if_exists='append', index=False)
            st.toast("Saved to database ✅")
        except Exception as e:
            st.error(f"Error during prediction/save: {e}")

    st.subheader("Current Data (latest 40 rows)")
    with get_conn() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE} ORDER BY rowid DESC LIMIT 40", conn)
    st.dataframe(df, use_container_width=True)

