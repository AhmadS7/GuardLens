import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from kafka import KafkaConsumer
import json
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes and edges
for _, row in data.iterrows():
    G.add_edge(row['Sender'], row['Receiver'], amount=row['Amount'])

# Compute centrality measures
centrality = nx.eigenvector_centrality(G)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 14  # Bottleneck layer size

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Compile model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train on normal transactions only
normal_data = X_train[y_train == 0]  # Non-fraudulent transactions
autoencoder.fit(normal_data, normal_data, epochs=50, batch_size=256, shuffle=True, validation_split=0.1)

# Detect anomalies
reconstruction_errors = np.mean(np.power(X_test - autoencoder.predict(X_test), 2), axis=1)
threshold = np.percentile(reconstruction_errors, 95)  # Set threshold at 95th percentile
predictions = (reconstruction_errors > threshold).astype(int)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequential data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 10
X_seq = create_sequences(X_train, seq_length)
y_seq = y_train[seq_length:]

# Define LSTM model
model = Sequential([
    LSTM(50, input_shape=(seq_length, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_seq, y_seq, epochs=20, batch_size=128, validation_split=0.1)

from sklearn.linear_model import LogisticRegression

# Simulate federated learning
clients = [X_train_1, X_train_2, X_train_3]  # Data from different clients
models = []

for client_data in clients:
    model = LogisticRegression()
    model.fit(client_data, y_train)
    models.append(model)

# Aggregate weights
global_model = LogisticRegression()
global_model.coef_ = np.mean([m.coef_ for m in models], axis=0)
global_model.intercept_ = np.mean([m.intercept_ for m in models], axis=0)

import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes and edges
for _, row in data.iterrows():
    G.add_edge(row['Sender'], row['Receiver'], amount=row['Amount'])

# Compute centrality measures
centrality = nx.eigenvector_centrality(G)

import tensorflow as tf

# Generate adversarial examples
epsilon = 0.01
with tf.GradientTape() as tape:
    tape.watch(X_test)
    predictions = model(X_test)
    loss = tf.keras.losses.binary_crossentropy(y_test, predictions)

gradients = tape.gradient(loss, X_test)
adversarial_examples = X_test + epsilon * tf.sign(gradients)from sklearn.linear_model import LogisticRegression

# Simulate federated learning
clients = [X_train_1, X_train_2, X_train_3]  # Data from different clients
models = []

for client_data in clients:
    model = LogisticRegression()
    model.fit(client_data, y_train)
    models.append(model)

# Aggregate weights
global_model = LogisticRegression()
global_model.coef_ = np.mean([m.coef_ for m in models], axis=0)
global_model.intercept_ = np.mean([m.intercept_ for m in models], axis=0)

import tensorflow as tf

# Generate adversarial examples
epsilon = 0.01
with tf.GradientTape() as tape:
    tape.watch(X_test)
    predictions = model(X_test)
    loss = tf.keras.losses.binary_crossentropy(y_test, predictions)

gradients = tape.gradient(loss, X_test)
adversarial_examples = X_test + epsilon * tf.sign(gradients)

from sklearn.ensemble import VotingClassifier

# Define individual models
models = [
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())
]

# Create voting classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_clf.fit(X_train, y_train)

from sklearn.ensemble import StackingClassifier

# Define base models and meta-model
base_models = [
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier())
]
meta_model = LogisticRegression()

# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_clf.fit(X_train, y_train)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName('FraudDetection').getOrCreate()

# Load data into Spark DataFrame
df = spark.read.csv('creditcard.csv', header=True, inferSchema=True)

# Train Random Forest model
rf = RandomForestClassifier(featuresCol='features', labelCol='Class')
model = rf.fit(df)



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