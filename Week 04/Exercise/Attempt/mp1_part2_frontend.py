"""
MP1 Part 2 - Streamlit Frontend for Physics Q&A
Frontend interface for the Physics Q&A chatbot.

Usage: streamlit run mp1_part2_frontend.py
"""

import streamlit as st
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="Physics Q&A Chatbot",
    page_icon="🔬",
    layout="wide"
)

# API endpoint
API_BASE_URL = "http://localhost:5000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ask_question(question):
    """Send question to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_contexts():
    """Get all available contexts from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/contexts", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# Main app
def main():
    st.title("🔬 Physics Q&A Chatbot")
    st.markdown("Ask me any physics question and I'll provide an answer based on physics concepts!")
    
    # Check API status
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Check API Status"):
            with st.spinner("Checking API..."):
                if check_api_health():
                    st.success("✅ API is running")
                else:
                    st.error("❌ API is not responding")
                    st.info("Make sure to run: `python mp1_part2_backend.py`")
    
    st.markdown("---")
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("📋 Example Questions")
        example_questions = [
            "What is quantum entanglement?",
            "What is the Third Law of Thermodynamics?",
            "What is Stefan number?",
            "What is the multi-verse theory?",
            "What is Pauli's exclusion principle?",
            "Explain Newton's First Law",
            "What is Ohm's Law?",
            "What is Hooke's Law?"
        ]
        
        st.markdown("Click on any example to try it:")
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.selected_question = question
        
        st.markdown("---")
        
        # Show available contexts
        if st.button("Show Available Physics Concepts"):
            with st.spinner("Loading contexts..."):
                contexts_data = get_contexts()
                if "error" not in contexts_data:
                    st.subheader("📚 Available Physics Concepts")
                    for i, context in enumerate(contexts_data.get("contexts", []), 1):
                        with st.expander(f"Concept {i}"):
                            st.write(context)
                else:
                    st.error(contexts_data["error"])
    
    # Main question input
    question_input = st.text_input(
        "🤔 Ask your physics question:",
        value=st.session_state.get("selected_question", ""),
        placeholder="e.g., What is Newton's Second Law?"
    )
    
    # Clear the selected question after use
    if "selected_question" in st.session_state:
        del st.session_state.selected_question
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    # Process question
    if ask_button and question_input.strip():
        with st.spinner("🤖 Thinking... This may take a moment..."):
            result = ask_question(question_input.strip())
        
        if "error" in result:
            st.error(f"❌ {result['error']}")
            if "Connection error" in result['error']:
                st.info("💡 Make sure the backend API is running: `python mp1_part2_backend.py`")
        else:
            # Display results
            st.success("✅ Answer Generated!")
            
            # Question
            st.subheader("❓ Question:")
            st.write(result.get('question', ''))
            
            # Answer
            st.subheader("💡 Answer:")
            st.write(result.get('answer', ''))
            
            # Retrieved contexts
            st.subheader("📖 Retrieved Physics Concepts:")
            contexts = result.get('retrieved_contexts', [])
            
            if contexts:
                for i, ctx in enumerate(contexts, 1):
                    with st.expander(f"Context {i} (Similarity: {ctx.get('similarity', 0):.3f})"):
                        st.write(ctx.get('context', ''))
            else:
                st.info("No contexts retrieved")
    
    elif ask_button and not question_input.strip():
        st.warning("⚠️ Please enter a question!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🔬 Physics Q&A Chatbot | Powered by TF-IDF Retrieval + T5 Language Model</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    