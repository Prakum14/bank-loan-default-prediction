import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path
import os

# Define the target variable
TARGET_COL = 'loan_paid_back'
ID_COL = 'id'

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom feature engineering steps to the input DataFrame.
    
    Args:
        df: Input DataFrame (train or test).
        
    Returns:
        DataFrame with new engineered features.
    """
    # 1. New Ratio Feature: Loan-to-Income Ratio
    # This feature often captures risk better than individual values
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-6) # Add small epsilon to avoid division by zero
    
    # 2. Extract Subgrade Category
    # The grade (A, B, C...) is the first letter of 'grade_subgrade'
    df['loan_grade'] = df['grade_subgrade'].str[0]
    
    # 3. Handle 'Employed' Status vs. Others
    # Create a binary feature for simple classification of employment risk
    df['is_employed'] = df['employment_status'].apply(
        lambda x: 1 if x in ['Employed', 'Self-employed'] else 0
    )
    
    return df

def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Creates and returns a scikit-learn ColumnTransformer for all preprocessing steps.
    This ensures transformations (like scaling and encoding) are consistent.
    
    Args:
        X_train: The training features DataFrame used to fit the transformers.
        
    Returns:
        A fitted ColumnTransformer object.
    """
    # Define feature types and their corresponding transformers
    
    # Numerical features to scale (after imputation)
    numeric_features = [
        'annual_income', 
        'debt_to_income_ratio', 
        'credit_score', 
        'loan_amount', 
        'interest_rate',
        'loan_to_income_ratio' # Engineered feature
    ]
    
    # Categorical features to one-hot encode
    categorical_features = [
        'gender', 
        'marital_status', 
        'education_level',
        'loan_purpose',
        'loan_grade' # Engineered feature
    ]
    
    # Numerical pipeline: Impute missing values with mean, then scale
    numerical_pipeline = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    # Categorical pipeline: Impute with 'missing', then One-Hot Encode
    categorical_pipeline = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    
    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', pd.Pipeline(numerical_pipeline), numeric_features),
            ('cat', pd.Pipeline(categorical_pipeline), categorical_features)
        ],
        remainder='drop', # Drop columns not specified in transformers
        verbose_feature_names_out=False # Ensures clean feature names in the output
    )
    
    # Fit the transformer on the training data
    preprocessor.fit(X_train)
    
    return preprocessor

def load_and_engineer_data(data_path: str):
    """
    Main function to load raw data, apply feature engineering, and prepare
    datasets for model training and prediction.
    
    Args:
        data_path: The path to the directory containing 'train.csv' and 'test.csv'.
        
    Returns:
        X_train, y_train, X_test, test_ids, preprocessor (tuple of processed data and transformer).
    """
    try:
        train_path = Path(data_path) / 'train.csv'
        test_path = Path(data_path) / 'test.csv'
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        print(f"Loaded training data: {df_train.shape}")
        print(f"Loaded test data: {df_test.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure 'train.csv' and 'test.csv' are in the '{data_path}' directory.")
        raise e
        
    # --- 1. Target and ID Extraction ---
    y_train = df_train[TARGET_COL]
    
    # Store IDs for submission file creation later
    train_ids = df_train[ID_COL]
    test_ids = df_test[ID_COL]

    # Drop the ID and target from the features sets
    X_train = df_train.drop(columns=[TARGET_COL, ID_COL])
    X_test = df_test.drop(columns=[ID_COL])
    
    # Drop the 'employment_status' and 'grade_subgrade' columns since we engineered features from them
    # and they are high cardinality or redundant after engineering
    X_train = X_train.drop(columns=['employment_status', 'grade_subgrade'])
    X_test = X_test.drop(columns=['employment_status', 'grade_subgrade'])
    
    # --- 2. Feature Engineering ---
    X_train_fe = feature_engineer(X_train)
    X_test_fe = feature_engineer(X_test)
    
    # --- 3. Create and Fit Preprocessor ---
    # The preprocessor is only fitted on the training data!
    preprocessor = create_preprocessor(X_train_fe)
    
    # --- 4. Apply Transformations ---
    # Apply the transformations to both train and test sets
    X_train_processed = preprocessor.transform(X_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)

    # Convert back to DataFrame (necessary if we want feature names, though model training doesn't strictly require it)
    # The ColumnTransformer in the preprocessor handles the data structure correctly for the model later.
    
    # For simplicity in this script, we return the NumPy arrays after transformation.
    print(f"Training features shape after processing: {X_train_processed.shape}")
    print(f"Test features shape after processing: {X_test_processed.shape}")
    
    return X_train_processed, y_train, X_test_processed, test_ids, preprocessor

if __name__ == '__main__':
    # Example usage when running this file directly
    X_train, y_train, X_test, test_ids, preprocessor = load_and_engineer_data('../../data/raw')
    print("\n--- Preprocessing Complete ---")
    print(f"X_train (first 5 rows):\n{X_train[:5, :]}")
