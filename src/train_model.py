import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv('../data/creditcard.csv')

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

# Save the model and scaler
joblib.dump(model, '../models/xgboost_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')