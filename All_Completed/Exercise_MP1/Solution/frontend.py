# frontend_app.py

import streamlit as st
import requests

API_QA = "http://127.0.0.1:5000/qa"  # your Flask /qa endpoint

st.set_page_config(page_title="Physics Q&A", layout="centered")
st.title("Physics Q&A")

q = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        try:
            r = requests.post(API_QA, json={"question": q, "top_k": 3}, timeout=60)
            if r.status_code == 200:
                st.subheader("Answer")
                st.write(r.json().get("answer", "").strip())
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
