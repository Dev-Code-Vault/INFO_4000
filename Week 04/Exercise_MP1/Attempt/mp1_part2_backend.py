# mp1_part2_backend.py
# part2 flask backend

#import libraries
from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

#create flask app
app = Flask(__name__)

#create bot
class PhysicsQABot:
    def __init__(self, context_file="physics_context.json"):
        #load contexts
        try:
            with open(context_file, "r") as f:
                data = json.load(f)
                self.contexts = [item["context"] for item in data]
        except:
            self.contexts = [
                "Newton's First Law: An object remains at rest or in uniform motion unless acted on by a force.",
                "Newton's Second Law: Force equals mass times acceleration (F=ma).",
                "Universal Gravitation: Every mass attracts every other mass with force proportional to mass and inverse square of distance.",
                "First Law of Thermodynamics: Energy cannot be created or destroyed, only transformed.",
                "Third Law of Thermodynamics: As temperature approaches absolute zero, entropy approaches a minimum.",
                "Quantum entanglement: particles share states regardless of distance.",
                "Pauli's exclusion principle: No two electrons in an atom share the same quantum numbers.",
                "Stefan-Boltzmann Law: Energy radiated by a black body is proportional to T⁴.",
                "Multiverse theory: Our universe is one of many universes."
            ]

        #setup retrieval
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.context_vectors = self.vectorizer.fit_transform(self.contexts)

        #setup model
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

    #retrieve context#
    def retrieve_context(self, question, top_k=2):
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.context_vectors).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [{"context": self.contexts[i], "similarity": float(sims[i])} for i in top_idx]

    #answer question
    def answer_question(self, question):
        contexts = self.retrieve_context(question)
        context_text = " ".join(c["context"] for c in contexts)
        prompt = f"Answer based on context: {context_text}\nQuestion: {question}\nAnswer:"
        ans = self.qa_pipeline(prompt, max_length=150)[0]["generated_text"].strip()
        return {"question": question, "answer": ans, "retrieved_contexts": contexts}

#init bot
bot = PhysicsQABot()

@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"})

#get answer
@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("question", "")
    if not q: return jsonify({"error": "Question required"}), 400
    return jsonify(bot.answer_question(q))

#get contexts
@app.route("/contexts", methods=["GET"])
def contexts(): return jsonify({"contexts": bot.contexts, "count": len(bot.contexts)})

#run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
