# 🇻🇳 Vietnam Air Quality Predictive System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Region](https://img.shields.io/badge/Region-Vietnam-red)]()
[![Focus](https://img.shields.io/badge/Focus-Hanoi%20%26%20HCMC-yellow)]()
[![Model](https://img.shields.io/badge/Best%20Model-Random%20Forest-green)]()

> **Project Goal:** An end-to-end data science pipeline to predict Air Quality Index (AQI) across Vietnam's major economic hubs, addressing the critical "North-South" pollution divide.

## 🚀 Key Features
* **Vietnam-Specific ETL:** Automated scraping of real-time sensors across 38 Vietnamese provinces from *AQI.in*.
* **Advanced Feature Engineering:**
    * **Cyclical Time:** Handled the continuity between 23:00 and 00:00 using Sine/Cosine encoding.
    * **Class Balancing:** Applied SMOTE to handle the scarcity of "Hazardous" events in Southern Vietnam.
* **Model Benchmarking:** Comparison of Linear Regression, Random Forest, XGBoost, and LSTM on local environmental data.

## 📊 Performance: Tree-Based vs Deep Learning
We benchmarked models on a location-stratified split to ensure geographic generalization across Vietnam.

| Model | RMSE (Lower is better) | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **1.27** | **0.56** | **0.999** |
| XGBoost | 1.25 | 0.67 | 0.999 |
| LSTM | 27.08 | 16.06 | 0.498 |

> *Key Finding:* Ensemble models (Random Forest) significantly outperformed LSTM. This proves that for this specific tabular dataset, robust feature engineering (rolling statistics, lag features) was more decisive than model complexity.

## 🔍 The "North-South" Pollution Divide
Our Exploratory Data Analysis (EDA) revealed distinct pollution fingerprints for Vietnam's regions:

### 1. Hanoi (The North) 🏭
* **Driver:** Particulate Matter ($PM_{2.5}$) accumulation.
* **Pattern:** Severe "U-shape" daily rhythm. Pollution peaks at night and drops between 12:00-17:00 due to the "Afternoon Ventilation" phenomenon.
* **Risk:** Unsafe air quality >77% of the time.

### 2. Ho Chi Minh City & Da Nang (The South/Center) 🌊
* **Driver:** Traffic-related Ozone ($O_3$) and photochemical smog.
* **Pattern:** Stable "Moderate" air quality (85-94% of the time).
* **Anomaly:** Da Nang shows the highest normalized Ozone scores despite low dust levels.

## 🛠 Tech Stack
* **Data Acquisition:** Custom Python Scraper (`BeautifulSoup`, `Requests`), PostgreSQL (`psycopg2`).
* **Data Science:** Pandas, NumPy, Scikit-learn (SMOTE for imbalance).
* **Forecasting:** XGBoost, Keras (LSTM).
* **Visualization:** Streamlit (Interactive Dashboard).

## 💻 How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/Miythra/vietnam-air-quality-forecasting.git](https://github.com/Miythra/vietnam-air-quality-forecasting.git)