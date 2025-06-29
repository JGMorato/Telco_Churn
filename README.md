# 📉 Telco Customer Churn Prediction

This project aims to analyze and predict customer churn in a telecommunications company using supervised machine learning techniques. It follows an end-to-end data science workflow including EDA, feature engineering, modeling, evaluation, and deployment readiness.

## 🔍 Dataset

- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customer records
- Features include demographic, service, and billing information

---

## 🧪 Project Structure
```
telco-churn/
│
├── data/
│ ├── raw/ # Original CSV file
│ ├── processed/ # Cleaned dataset after EDA
│ └── predicted_from_test.csv # Predictions from X_test
│
├── models/
│ ├── random_forest_model.pkl # Trained pipeline
│ ├── threshold.json # Optimal decision threshold
│ └── X_test_sample.joblib # Sample test data
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ └── 02_modeling.ipynb # Model training and tuning
│
├── predict_churn.py # Prediction function
├── run_prediction.py # Example prediction script
├── requirements.txt # Python dependencies
└── README.md
```

## ⚙️ Key Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP (tentado, mas não usado)
- Jupyter Notebooks
- Git & GitHub

---

## 📈 Model Performance

- **Best model:** Tuned Random Forest  
- **Test ROC AUC:** 0.8361  
- **Test F1 Score:** 0.6313  
- **Threshold optimized for recall and F1**

---

## 🧠 Highlights

- Extensive EDA with visualizations and insights
- Feature engineering and binning for tenure
- Hyperparameter tuning with RandomizedSearchCV
- Threshold adjustment for better business recall
- Deployment-ready scripts for new customer prediction

---

## 🚀 How to Run

1. Create virtual environment  
   `python -m venv .venv`  
2. Activate it and install dependencies  
   `pip install -r requirements.txt`  
3. Run predictions on sample data  
   `python run_prediction.py`

---

## 📦 Dataset

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 👤 Author

João Gabriel Morato  
