"""
MP1 Part 2 - Physics Q&A Backend API
Context retrieval-based Physics Q&A chatbot backend using TF-IDF and T5.

Usage: python mp1_part2_backend.py
"""

from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

class PhysicsQABot:
    def __init__(self, context_file="physics_context.json"):
        self.contexts = []
        self.vectorizer = None
        self.context_vectors = None
        self.qa_pipeline = None
        
        # Load physics contexts
        self.load_contexts(context_file)
        
        # Initialize TF-IDF vectorizer
        self.setup_retrieval()
        
        # Initialize T5 model for Q&A
        self.setup_qa_model()
    
    def load_contexts(self, context_file):
        """Load physics contexts from JSON file"""
        try:
            with open(context_file, 'r') as f:
                data = json.load(f)
                self.contexts = [item['context'] for item in data]
            print(f"✅ Loaded {len(self.contexts)} physics contexts")
        except Exception as e:
            print(f"❌ Error loading contexts: {e}")
            # Fallback contexts if file not found
            self.contexts = [
                "Newton's First Law of Motion states that an object will remain at rest or in uniform motion unless acted upon by an external force.",
                "Newton's Second Law states that the force acting on an object is equal to the mass of that object multiplied by its acceleration (F = ma).",
                "The Law of Universal Gravitation states that every mass attracts every other mass with a force proportional to their masses and inversely proportional to the square of the distance between them.",
                "The First Law of Thermodynamics states that energy cannot be created or destroyed, only transferred or changed in form.",
                "Quantum entanglement is a phenomenon where particles become interconnected and the quantum state of one particle instantaneously affects another, regardless of distance.",
                "Pauli's exclusion principle states that no two electrons in an atom can have the same set of quantum numbers."
            ]
    
    def setup_retrieval(self):
        """Setup TF-IDF vectorizer for context retrieval"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Fit and transform contexts
        self.context_vectors = self.vectorizer.fit_transform(self.contexts)
        print("✅ TF-IDF vectorizer initialized")
    
    def setup_qa_model(self):
        """Setup T5 model for question answering"""
        try:
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=150,
                do_sample=True,
                temperature=0.7
            )
            print("✅ T5 model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading T5 model: {e}")
    
    def retrieve_context(self, question, top_k=2):
        """Retrieve most relevant contexts using TF-IDF similarity"""
        # Transform question to vector
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.context_vectors).flatten()
        
        # Get top-k most similar contexts
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_contexts = []
        for idx in top_indices:
            retrieved_contexts.append({
                'context': self.contexts[idx],
                'similarity': float(similarities[idx])
            })
        
        return retrieved_contexts
    
    def generate_answer(self, question, contexts):
        """Generate answer using T5 model with retrieved contexts"""
        # Combine contexts
        context_text = " ".join([ctx['context'] for ctx in contexts])
        
        # Create prompt for T5
        prompt = f"Answer the question based on the context: {context_text}\n\nQuestion: {question}\nAnswer:"
        
        # Generate answer
        try:
            result = self.qa_pipeline(prompt, max_length=150, num_return_sequences=1)
            answer = result[0]['generated_text'].strip()
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question):
        """Main function to answer physics questions"""
        # Retrieve relevant contexts
        contexts = self.retrieve_context(question)
        
        # Generate answer
        answer = self.generate_answer(question, contexts)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_contexts': contexts
        }

# Initialize the bot
physics_bot = PhysicsQABot()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Physics Q&A API is running'})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking physics questions"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get answer from the bot
        response = physics_bot.answer_question(question)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/contexts', methods=['GET'])
def get_contexts():
    """Get all available physics contexts"""
    return jsonify({'contexts': physics_bot.contexts, 'count': len(physics_bot.contexts)})

if __name__ == '__main__':
    print("🚀 Starting Physics Q&A API...")
    print("📚 Available endpoints:")
    print("  - POST /ask - Ask a physics question")
    print("  - GET /health - Health check")
    print("  - GET /contexts - View all contexts")
    print("\n🌐 API running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

    