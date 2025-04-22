import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from kafka import KafkaConsumer
import json

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

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display basic information
print(data.head())
print(data.info())
print(data['Class'].value_counts())  # Check class distribution (0 = non-fraud, 1 = fraud)

# Separate features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# Initialize and train the model
model = XGBClassifier(scale_pos_weight=100)  # Adjust weight for imbalanced data
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Example: Retrain with new data
new_data = pd.read_csv('new_transactions.csv')
X_new, y_new = preprocess(new_data)
model.fit(X_new, y_new)


# Explain model predictions
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)