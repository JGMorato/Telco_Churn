import joblib
import json
import pandas as pd

# Load model and threshold
model = joblib.load('models/random_forest_model.pkl')

with open('models/threshold.json', 'r') as f:
    threshold = json.load(f)['threshold']

def predict_churn(input_data):
    """
    Predict churn from new data using the trained model and chosen threshold.
    
    Parameters:
    - input_data: pd.DataFrame with same structure as training features (before encoding)

    Returns:
    - pd.DataFrame with original data + 'Churn_Probability' and 'Churn_Predicted'
    """
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Predict churn probability (class 1)
    proba = model.predict_proba(input_data)[:, 1]

    # Apply threshold
    prediction = (proba >= threshold).astype(int)

    # Return with predictions
    result = input_data.copy()
    result['Churn_Probability'] = proba
    result['Churn_Predicted'] = prediction

    return result