import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# LOAD DATASET
data = pd.read_csv(r"C:\Users\Priya\Downloads\INDIA_AQI_CLEANED.csv")

# SELECT FEATURES
X = data[['PM2_5_ugm3','PM10_ugm3','NO2_ugm3','SO2_ugm3']]
y = data['AQI_Category']

# ENCODE TARGET
le = LabelEncoder()
y = le.fit_transform(y)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(model, "aqi_health_model.pkl")

print("Model Trained Successfully")

# PREDICTION
y_pred = model.predict(X_test)

# EVALUATION METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Evaluation Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# USER INPUT
pm25 = float(input("\nEnter PM2.5 value: "))
pm10 = float(input("Enter PM10 value: "))
no2 = float(input("Enter NO2 value: "))
so2 = float(input("Enter SO2 value: "))

input_data = np.array([[pm25, pm10, no2, so2]])

prediction = model.predict(input_data)
result = le.inverse_transform(prediction)

print("\nPredicted AQI Category:", result[0])