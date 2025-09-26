# exercise1.py
"""I used tabs to separate inot different veiws, and I
   used columns to separate into two sections. And the Expander
   is used to show instructions for the usesr."""

# import necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# set page configuration
st.set_page_config(layout="wide")

# app title and caption
st.title("Streamlit Exercise")
st.caption("A basic UI")

# create 2 columns
left_col, right_col = st.columns(2)

# left column
with left_col:
    # instructions for the user
    with st.expander("Instructions"):
        st.markdown("""
        - Enter your name.
        - Choose how many random data points to generate.
        - Optionally upload an image.
        - Click "Run" to view results on the right.
        """)

    name = st.text_input("Your Name")
    n_points = st.number_input("Number of Random Points ", min_value=1, max_value=1000, value=10)
    image_file = st.file_uploader("Upload an image ", type=["png", "jpg", "jpeg"])
    run_button = st.button("Run")

# right column
with right_col:
    if run_button:
        with st.container():
            # results section
            tab1, tab2 = st.tabs(["Summary", "Details"])

            with tab1:
                if name:
                    st.success(f"Hello, **{name}**! 👋")
                else:
                    st.info("No name provided.")

                #generate random data and the plot
                random_data = np.random.randn(int(n_points))
                fig, ax = plt.subplots()
                ax.plot(random_data, marker='o')
                ax.set_title("Random Data Chart")
                st.pyplot(fig)

                #display image
                if image_file:
                    st.image(image_file, caption="Uploaded Image", use_column_width=True)
                else:
                    st.write("No image uploaded.")

            with tab2:
                st.code(f"Generated {int(n_points)} random points.")
                if image_file:
                    st.write("Image file name:", image_file.name)

