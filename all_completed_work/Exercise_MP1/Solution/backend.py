# app.py
# Single-route Physics Q&A backend (retrieval + generation)

from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
from hf_access import ACCESS_TOKEN
from huggingface_hub import login

# Your Hugging Face API token (get it from hf.co/settings/tokens)
# It's best practice to store this in a separate file or as a Streamlit secret or environment variable
HF_TOKEN = ACCESS_TOKEN
login(HF_TOKEN)

# Load corpus and create knowledge base
with open("physics_context.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

docs = [item["context"] for item in corpus]   # one key: "context"

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",  # <-- allow 1-char tokens like F, Q, V because physic could have such terms
    ngram_range=(1, 2)             # <-- unigrams + bigrams for equations like F=ma
)
tdm = vectorizer.fit_transform(docs)  # Create the embeddings

#  HF model  
device = 0 if torch.cuda.is_available() else -1
gen = pipeline("text2text-generation", model="google/flan-t5-small", device=device)

# Flask
app = Flask(__name__)

@app.post("/qa")
def qa():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k") or 3)
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # retrieval
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tdm)[0]
    idxs = sims.argsort()[::-1][:top_k]
    context = "\n\n".join([docs[i] for i in idxs])

    # generation
    prompt = (
        "You are a helpful physics tutor. Use ONLY the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    out = gen(prompt, max_new_tokens=120, do_sample=True, temperature=0.7)

    return jsonify({"answer": out[0]["generated_text"]})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
