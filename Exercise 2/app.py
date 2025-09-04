# app.py

#import libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib
import sqlite3
import pandas as pd

#load model
app = Flask(__name__)
model = joblib.load("model.joblib")

#predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)

        #save data to the database
        if "team" in data:
            conn = sqlite3.connect("NFL.db")
            new_row = pd.DataFrame([{
                "Team": data["team"],
                "PF": features[0][0],
                "PA": features[0][1],
                "PD": features[0][2],
                "SoS": features[0][3],
                "WinningSeason": int(prediction[0])
            }])
            new_row.to_sql("stats", conn, if_exists="append", index=False)
            conn.close()
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

#run
if __name__ == '__main__':
    app.run(port=5000, debug=True)
