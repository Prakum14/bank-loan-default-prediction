import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# --- Define Paths ---
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'final_model.pkl')
PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, 'test_predictions.csv')
REPORT_PATH = os.path.join(ARTIFACTS_DIR, 'training_report.txt')

def train_pipeline():
    # 1. Ensure the artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"Artifacts directory created at: {ARTIFACTS_DIR}")

    # 2. Load Data (assuming LFS successfully pulled train.csv)
    try:
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        print(f"Data loaded. Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure 'train.csv' and 'test.csv' are in the 'data' directory.")
        return

    # 3. Simple Feature Engineering / Preprocessing
    # Select only numeric features for simplicity and handle missing values
    numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    X = df_train[numeric_features].fillna(df_train[numeric_features].median())
    y = df_train['loan_paid_back']
    
    X_test = df_test[numeric_features].fillna(df_test[numeric_features].median())

    # 4. Train the Model
    # Using a simple Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    print("Model training complete.")

    # 5. Evaluate and Predict
    train_pred_proba = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, train_pred_proba)
    
    test_predictions = model.predict(X_test)
    
    # 6. Save Artifacts
    
    # a. Save the model object
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # b. Save the test predictions
    submission_df = pd.DataFrame({
        'id': df_test['id'],
        'loan_paid_back': test_predictions.astype(int)
    })
    submission_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    
    # c. Save a simple report
    with open(REPORT_PATH, 'w') as f:
        f.write("--- Model Training Report ---\n")
        f.write(f"Model: Logistic Regression\n")
        f.write(f"Trained on {df_train.shape[0]} samples.\n")
        f.write(f"Training AUC Score: {train_auc:.4f}\n")
        f.write(f"Features used: {', '.join(numeric_features)}\n")
        f.write("-----------------------------\n")
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    train_pipeline()
