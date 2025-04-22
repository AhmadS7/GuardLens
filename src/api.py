from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('../models/xgboost_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    features = np.array(input_data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)