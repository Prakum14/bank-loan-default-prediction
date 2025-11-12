import os
import pandas as pd
import pytest

# Define expected data paths relative to the project root
TRAIN_DATA_PATH = os.path.join("data", "train.csv")
TEST_DATA_PATH = os.path.join("data", "test.csv")
ARTIFACTS_DIR = "artifacts"

# --- Test Data Integrity (Sanity Checks) ---

def test_data_files_exist():
    """Checks if the train and test data files are present in the 'data' directory."""
    assert os.path.exists(TRAIN_DATA_PATH)
    assert os.path.exists(TEST_DATA_PATH)

def test_train_data_expected_columns():
    """
    Checks if the training data has the necessary columns to run the training script.
    This ensures data schema consistency.
    """
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        pytest.skip(f"Training data not found at {TRAIN_DATA_PATH}. Skipping schema test.")

    # The expected list of columns based on the dataset schema
    expected_cols = [
        'id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 
        'loan_amount', 'interest_rate', 'gender', 'marital_status', 
        'education_level', 'employment_status', 'loan_purpose', 
        'grade_subgrade', 'loan_paid_back'
    ]
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    assert not missing_cols, f"Training data is missing critical columns: {missing_cols}"

def test_target_variable_is_binary():
    """Checks that the target variable ('loan_paid_back') only contains binary values (0 or 1)."""
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        pytest.skip("Training data not found. Skipping target check.")
        
    target_values = df['loan_paid_back'].dropna().unique()
    valid_values = {0.0, 1.0}
    
    assert set(target_values).issubset(valid_values), \
        f"Target variable ('loan_paid_back') has unexpected values: {target_values}"


# --- Test Artifacts Integrity ---

def test_artifacts_directory_exists():
    """Ensures the directory for saving the model and preprocessor exists."""
    assert os.path.isdir(ARTIFACTS_DIR), f"Artifacts directory '{ARTIFACTS_DIR}' does not exist."
