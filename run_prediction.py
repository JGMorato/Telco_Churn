import pandas as pd
from predict_churn import predict_churn
from joblib import load

# Load X_test from the modeling notebook
X_test = load('models/X_test_sample.joblib')  # ou outro caminho

# Run prediction
result = predict_churn(X_test)

# Show results
print(result[['Churn_Probability', 'Churn_Predicted']].head())

# Saving results
result.to_csv('data/predicted_from_test.csv', index=False)