from kafka import KafkaConsumer
import json
import joblib

# Load the trained model and scaler
model = joblib.load('../models/xgboost_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Process incoming transactions
for message in consumer:
    transaction = message.value['features']
    scaled_features = scaler.transform([transaction])
    prediction = model.predict(scaled_features)
    print(f"Transaction: {transaction}, Prediction: {prediction}")