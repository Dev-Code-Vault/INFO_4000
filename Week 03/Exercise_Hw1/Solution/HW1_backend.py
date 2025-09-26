from flask import Flask, request, jsonify
import os
import torch
from HW1_backend_class import MyPredictor
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create an instance of the class
predictor = MyPredictor()

# Flask route for predicting and returning prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the model name and image type from the request form data
    model_name = request.form.get('model_name')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image from the in-memory file stream
        img_bytes = file.read()
        
        # Convert the byte stream to a PIL Image object
        image = Image.open(io.BytesIO(img_bytes))

        # Perform the prediction with the specified image type
        result, confidence = predictor.predict(image=image, model_type=model_name)

        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

