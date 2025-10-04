# app/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("../iris_pipeline.joblib")  # تأكدي من المسار الصحيح

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # نتوقع JSON مثل: {"features": [5.1, 3.5, 1.4, 0.2]}
    features = np.array(data['features']).reshape(1, -1)
    pred = model.predict(features)[0]
    probs = model.predict_proba(features).tolist()[0]
    return jsonify({"prediction": int(pred), "probabilities": probs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

