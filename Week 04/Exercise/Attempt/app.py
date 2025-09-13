from flask import Flask, request, jsonify
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

app = Flask(__name__)

# ✅ Load model and processor from local 'results/' folder
model = AutoModelForImageClassification.from_pretrained("results")
processor = AutoImageProcessor.from_pretrained("results")

classifier = pipeline("image-classification", model=model, feature_extractor=processor)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    preds = classifier(image)
    return jsonify(preds)

if __name__ == "__main__":
    app.run(debug=True)
