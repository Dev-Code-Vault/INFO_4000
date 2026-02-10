# mp1_part2_frontend.py
# part2 streamlit frontend
# to run do : streamlit run mp1_part2_frontend.py

# importing libraries
import streamlit as st
import requests

# API endpoint
API_BASE_URL = "http://localhost:5000"

# function to send question to the API
def ask_question(question):
    """Send question to the API"""
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# main function
def main():
    st.set_page_config(page_title="Physics Q&A", page_icon="🔬")
    st.title("🔬 Physics Q&A Chatbot")
    st.markdown("Ask a physics question and get an answer powered by TF-IDF retrieval + T5 generation.")

    #input box
    question = st.text_input("Enter your physics question:", placeholder="e.g., What is Newton's Second Law?")

    #buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    with col2:
        if st.button("Clear"):
            st.rerun()

    #procesing the question
    if ask_button and question.strip():
        with st.spinner("Generating answer..."):
            result = ask_question(question.strip())

        if "error" in result:
            st.error(result["error"])
            st.info("Make sure the backend API is running with: python mp1_part2_backend.py")
        else:
            st.subheader("Answer")
            st.write(result.get("answer", ""))

            # show contexts if possible
            contexts = result.get("retrieved_contexts", [])
            if contexts:
                st.subheader("Retrieved Physics Concepts")
                for i, ctx in enumerate(contexts, 1):
                    st.write(f"**Context {i}:** {ctx.get('context', '')}")

    # error handling
    elif ask_button and not question.strip():
        st.warning("Please enter a question first")

    # footer
    st.markdown("---")
    st.caption("Physics Q&A System | MP1 Part 2")


if __name__ == "__main__":
    main()
