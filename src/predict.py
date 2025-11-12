mport pandas as pd
import joblib
import os
import numpy as np

# --- Configuration ---
# Define the directory where the model and preprocessor artifacts are stored.
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')

def load_artifacts():
    """Loads the trained model and the ColumnTransformer (preprocessor)."""
    try:
        # Load the fitted preprocessor
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Preprocessor loaded successfully from {PREPROCESSOR_PATH}")

        # Load the trained model
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")

        return model, preprocessor
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required artifact file. Have you run 'python src/train.py' yet?")
        print(f"Missing file: {e.filename}")
        raise

def predict_new_data(new_loan_data: dict, model, preprocessor) -> dict:
    """
    Makes a prediction on a single new loan application using loaded artifacts.

    Args:
        new_loan_data: A dictionary containing the features of the new loan applicant.
        model: The loaded trained machine learning model.
        preprocessor: The loaded fitted ColumnTransformer.

    Returns:
        A dictionary containing the prediction (loan_paid_back) and probability.
    """
    # 1. Convert the input dictionary to a DataFrame (crucial for ColumnTransformer)
    # We use a list containing the dictionary to ensure the DataFrame has one row.
    data_df = pd.DataFrame([new_loan_data])

    # 2. Separate features from the ID (if present in the input, which it isn't here, 
    # but good practice is to treat the DataFrame as expected by the preprocessor)
    X_new = data_df 

    # 3. Apply the fitted preprocessor
    # Note: ColumnTransformer output is typically a numpy array
    X_processed = preprocessor.transform(X_new)

    # 4. Make prediction
    # Predict the class (0 or 1)
    prediction = model.predict(X_processed)[0]

    # Predict the probability of the positive class (1: paid back)
    probability = model.predict_proba(X_processed)[0, 1]

    # 5. Format and return results
    result = {
        'input_data': new_loan_data,
        'prediction_class': int(prediction),
        'probability_of_paying_back': round(probability, 4)
    }

    return result

if __name__ == '__main__':
    print("--- Starting ML Inference Script (src/predict.py) ---")
    
    # Load the necessary artifacts
    try:
        loaded_model, loaded_preprocessor = load_artifacts()
    except Exception as e:
        print("Inference aborted due to missing artifacts.")
        # Exit gracefully if artifacts can't be loaded
        exit(1)

    # --- Sample New Loan Application Data ---
    # This data structure must match the features used during training.
    # Data is for a fictional new applicant.
    sample_data = {
        'id': 999999,
        'annual_income': 65000.00,
        'debt_to_income_ratio': 0.15,
        'credit_score': 720,
        'loan_amount': 15000.00,
        'interest_rate': 8.5,
        'gender': 'Male',
        'marital_status': 'Married',
        'education_level': "Master's",
        'employment_status': 'Employed',
        'loan_purpose': 'Home',
        'grade_subgrade': 'B2'
    }

    print("\nAttempting prediction for sample data:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
        
    # Make and print the prediction
    prediction_result = predict_new_data(sample_data, loaded_model, loaded_preprocessor)

    print("\n--- Prediction Result ---")
    if prediction_result['prediction_class'] == 1:
        print("🟢 Prediction: Loan is likely to be **PAID BACK** (Class 1)")
    else:
        print("🔴 Prediction: Loan is likely to **DEFAULT** (Class 0)")
        
    print(f"Probability of being paid back: {prediction_result['probability_of_paying_back'] * 100:.2f}%")
    print("-------------------------")
    print(f"Full Result Dictionary: {prediction_result}")
