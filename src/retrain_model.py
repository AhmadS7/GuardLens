import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load new data
new_data = pd.read_csv('../data/new_transactions.csv')

# Preprocess new data
def preprocess(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    scaler = joblib.load('../models/scaler.pkl')
    X_scaled = scaler.transform(X)
    return X_scaled, y

X_new, y_new = preprocess(new_data)

# Retrain the model
model = joblib.load('../models/xgboost_model.pkl')
model.fit(X_new, y_new)

# Save the updated model
joblib.dump(model, '../models/xgboost_model.pkl')