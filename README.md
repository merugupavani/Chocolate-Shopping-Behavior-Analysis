# 🍫 Chocolate Customer Behavior Predictor

An end-to-end Machine Learning application that predicts customer spending categories (Low, Medium, or High) based on shopping behavior and demographic data. Developed in **Jupyter Notebook** and deployed as an interactive web dashboard using **Streamlit**.

## 🚀 Project Overview
This project analyzes how factors like age, seasonal trends, and discount usage influence chocolate purchasing habits. By utilizing a trained classification model, businesses can predict if a customer will be a "High Spender," allowing for targeted marketing and inventory management.

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Data Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Random Forest/XGBoost), Pickle
*   **Frontend/Deployment:** Streamlit
*   **Environment:** Jupyter Notebook

## 📂 Project Structure
```text
├── data/                   # Dataset files (CSV)
├── notebooks/              # Jupyter Notebooks for EDA and Modeling
├── app.py                  # Streamlit application script
├── model.pkl               # Saved trained ML model
├── scaler.pkl              # Saved StandardScaler object
├── features.pkl            # List of feature names for encoding alignment
└── requirements.txt        # List of dependencies
