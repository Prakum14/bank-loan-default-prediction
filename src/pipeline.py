import pandas as pd
from pathlib import Path
import os
import sys

# Ensure the 'src' directory is in the path to allow local imports
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import load_and_engineer_data
from model_trainer import (
    train_model, 
    evaluate_model, 
    make_predictions, 
    save_artifacts
)

# --- Configuration ---
DATA_PATH = Path('data') / 'raw'
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

SUBMISSION_FILENAME = 'loan_prediction_submission.csv'
ARTIFACTS_FILENAME = 'model_artifacts.joblib'


def create_submission_file(test_ids: pd.Series, predictions: pd.Series):
    """
    Creates the final submission CSV file.
    """
    print("\n--- Creating Submission File ---")
    submission_df = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': predictions.round(4)
    })
    submission_path = OUTPUT_DIR / SUBMISSION_FILENAME
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved successfully to: {submission_path}")


def run_pipeline():
    """
    The main function to execute the entire machine learning pipeline.
    """
    print("--- Starting Loan Default Prediction ML Pipeline ---")
    
    X_train, y_train, X_test, test_ids, preprocessor = load_and_engineer_data(DATA_PATH)
    
    model = train_model(X_train, y_train)
    
    roc_auc_score = evaluate_model(model, X_train, y_train)
    
    test_predictions = make_predictions(model, X_test)
    
    save_artifacts(model, preprocessor, ARTIFACTS_FILENAME, roc_auc_score)
    create_submission_file(test_ids, test_predictions)
    
    print("\n--- Pipeline Execution Complete! ---")


if __name__ == '__main__':
    run_pipeline()
