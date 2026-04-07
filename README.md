# Minor_Project-PR1103-
# AI-Based Air Quality and Health Risk Monitoring System

##  Project Overview

This project presents an **AI-powered dashboard** for monitoring air quality and predicting future pollution levels. It combines **data analytics, machine learning (LSTM), and interactive visualization** to provide meaningful insights into environmental conditions and health risks.

The system not only analyzes historical AQI data but also predicts **future Air Quality Index (AQI)** trends, making it a **predictive analytics solution** rather than just a static dashboard.

---

##  Objectives

* Monitor air quality across different cities
* Analyze environmental and weather factors affecting AQI
* Predict future AQI using **LSTM (Long Short-Term Memory)**
* Estimate **Health Risk Levels** based on multiple parameters
* Build an **interactive dashboard** for visualization

---

##  Key Features

### Interactive Dashboard (Streamlit)

* City and Year filtering
* Real-time data visualization
* KPI indicators (Average AQI, Health Risk Score, etc.)

###  Machine Learning Integration

* LSTM model trained on historical AQI data
* Captures temporal patterns in pollution data
* Predicts future AQI values

###  Data Visualization

* AQI trends over time
* Wind Speed vs AQI
* Solar Radiation impact
* Hourly AQI variation

###  Future Prediction

* Predicts next 10 time steps of AQI
* Combines past + predicted values in a single graph

###  Health Risk Analysis

* Custom Health Risk Score calculation
* Categorized into:

  * Low Risk
  * Moderate Risk
  * High Risk
  * Severe Risk

---

##  Machine Learning Approach

### Model Used:

**LSTM (Long Short-Term Memory)**

### Why LSTM?

* Handles sequential/time-series data
* Learns patterns from historical AQI trends
* Suitable for forecasting future values

### Working:

1. Takes last 10 AQI values as input
2. Learns patterns using neural networks
3. Predicts next AQI value
4. Repeats recursively for future forecasting

---

##  Tech Stack

| Category        | Tools Used         |
| --------------- | ------------------ |
| Programming     | Python             |
| Dashboard       | Streamlit          |
| ML Framework    | TensorFlow / Keras |
| Data Processing | Pandas, NumPy      |
| Visualization   | Plotly             |
| Model Saving    | Joblib             |

---

##  Project Structure

```
 project-folder
│── app.py                  # Streamlit Dashboard
│── lstm.py                 # Model training script
│── lstm_aqi_model.h5       # Trained ML model
│── scaler.save             # Data scaler
│── INDIA_AQI_CLEANED.csv   # Dataset
│── README.md               # Documentation
```

---

## Installation & Setup
## Clone Repository

###  Install Dependencies

```
pip install -r requirements.txt
```

###  Run Dashboard

```
streamlit run app.py
```

---

##  Dataset

* Contains AQI and environmental parameters
* Includes:

  * AQI values
  * Temperature
  * Humidity
  * Wind Speed
  * Rainfall
  * Solar Radiation
  * Pressure

---

##  Results

* Model successfully trained using LSTM
* RMSE achieved: ~110 (acceptable for project-level forecasting)
* Dashboard provides both:

  * Analytical insights
  * Predictive capabilities

---

##  Future Scope

* Integration with **real-time AQI APIs**
* More advanced models (Hybrid / Deep Learning)
* Mobile application deployment
* Health recommendation system
* Multi-city real-time comparison

---

##  Learning Outcomes

* Understanding of time-series forecasting
* Hands-on experience with LSTM models
* Building end-to-end ML applications
* Dashboard development using Streamlit
* Data preprocessing and feature engineering

---

##  Author

**Priyanshi Mehta**

---

##  Conclusion

This project demonstrates how **Artificial Intelligence can be used to transform environmental data into actionable insights**. By integrating machine learning with visualization, the system moves from **descriptive analytics to predictive intelligence**, making it highly relevant for smart city applications.

---



The given link is of dataset used in this project:
https://drive.google.com/file/d/1uk0XHNkWyjD3yYgd2coZOnFladmpBvcX/view?usp=drive_link
