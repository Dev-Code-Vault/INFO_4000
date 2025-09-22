"""
Simple Streamlit UI that talks to the local FastAPI backend.
Run:
    streamlit run part2_streamlit_frontend.py
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/answer"

st.title("Physics Q&A — Retrieval + LLM (Simple)")

question = st.text_area("Enter your physics question", height=120)
k = st.slider("Number of contexts to retrieve (k)", 1, 3, 1)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Getting answer..."):
            resp = requests.post(API_URL, json={"question": question, "k": k}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Answer")
                st.write(data["answer"])
                st.subheader("Retrieved context(s)")
                for r in data["retrieved"]:
                    st.markdown(f"**{r.get('title','(no title)')}** — score: {r['score']:.4f}")
                    st.write(r["context"])
            else:
                st.error(f"API error: {resp.status_code} - {resp.text}")
