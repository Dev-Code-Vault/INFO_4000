# Exercise 1 Solution

# There are many ways to develop this and the idea was to learn to use streamlit to develop frontends

import streamlit as st, numpy as np, pandas as pd

st.set_page_config(page_title="Mini Dashboard", layout="wide")
st.title("✨ Mini Dashboard")
st.caption("Practice columns, containers, and tabs.")

# --- Layout: inputs left, outputs right ---
left, right = st.columns([1, 2], gap="large")

with left:
    with st.expander("📖 Instructions", expanded=True):
        st.write(
            "1) Enter your name and N.\n2) (Optional) Upload an image.\n3) Click Run."
        )
    name = st.text_input("Your name")
    N = st.number_input(
        "Number of points (N)", min_value=10, max_value=500, value=100, step=10
    )
    img_file = st.file_uploader(
        "Upload an image (optional)", type=["png", "jpg", "jpeg"]
    )
    run = st.button("Run")

with right:
    tabs = st.tabs(["Summary", "Details"])
    results = st.container()  # <-- main output area

# --- Interactions: populate the results container when Run is clicked ---
if run:
    with results:
        if name:
            st.success(f"Hello, **{name}**! 👋")
        data = np.random.randn(int(N), 1)
        st.line_chart(data)

        if img_file:
            st.image(img_file, caption="Uploaded image", use_container_width=True)

    with tabs[0]:
        st.metric("Points plotted", int(N))
    with tabs[1]:
        df = pd.DataFrame(data, columns=["value"])
        st.dataframe(df.head(), use_container_width=True)
