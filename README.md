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

## 📊 Main Steps

### 1. Exploratory Data Analysis (EDA)
- Categorical feature distribution and churn rates
- Numerical variable patterns by churn (tenure, charges)
- Correlation analysis
- Outlier inspection and treatment strategy

### 2. Feature Engineering
- Binning of `tenure`
- Derived features like `charges_per_month`
- One-hot encoding for categorical variables

### 3. Modeling
- Baseline: Logistic Regression
- Random Forest with hyperparameter tuning (`RandomizedSearchCV`)
- Threshold optimization based on F1-Score
- ROC AUC used as main evaluation metric

### 4. Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.7932   | 0.6334    | 0.5267 | 0.5752   | 0.8346  |
| Random Forest (tuned) | 0.7910 | 0.6429    | 0.4813 | 0.5505   | 0.8361  |
| Random Forest (threshold 0.35) | 0.7726 | 0.5547 | 0.7326 | 0.6313 | 0.8361 |

---

## 🔮 Deployment Simulation

### ✅ Predict churn from new data

The script `predict_churn.py` loads the trained pipeline and chosen threshold to classify churn for new customer data.

### ▶️ How to run:

```bash
# Step 1: Activate your environment (if applicable)
# Step 2: Run the script
python run_prediction.py
```

The script will:

- Load X_test_sample.joblib as sample input

- Predict churn probabilities and labels

- Save results in data/predicted_from_test.csv

💾 Requirements
To install dependencies:

```bash
pip install -r requirements.txt
```

📁 Files to Explore
- 01_eda.ipynb: Full data analysis and preprocessing steps

- 02_modeling.ipynb: Model training, evaluation, and tuning

- predict_churn.py: Prediction script for unseen data

- run_prediction.py: Example script for batch prediction

- models/: Contains trained model and decision threshold

🚀 Next Steps (Future Work)
- Try SHAP for interpretability (currently incompatible with pipeline setup)

- Export predictions to a web app (e.g., Streamlit or Flask)

- Compare with gradient boosting (XGBoost, LightGBM) after tuning

👨‍💻 Author
João Gabriel Morato
Data Scientist — Portfolio Project
