import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from pathlib import Path

# --- Configuration ---
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILENAME = 'default_prediction_model.joblib'

def train_model(X_train: np.ndarray, y_train: pd.Series):
    """
    Initializes and trains a Random Forest Classifier with balanced class weights.
    
    Args:
        X_train: The processed training features (NumPy array).
        y_train: The training target variable (Pandas Series).
        
    Returns:
        The trained Random Forest model.
    """
    print("Initializing Random Forest Classifier...")
    # Use class_weight='balanced' to handle potential imbalance in loan defaults
    # n_estimators and max_depth are set moderately for quick, yet effective training
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1 # Use all available CPU cores for speed
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    return model

def evaluate_model(model, X_data: np.ndarray, y_true: pd.Series) -> float:
    """
    Evaluates the trained model and returns the ROC AUC score.
    
    Args:
        model: The trained model object.
        X_data: The feature data to evaluate on.
        y_true: The true labels.
        
    Returns:
        The ROC AUC score achieved on the data.
    """
    # Predict probabilities for the positive class (1.0 - paid back)
    y_proba = model.predict_proba(X_data)[:, 1]
    
    # Predict class labels for other metrics
    y_pred = model.predict(X_data)
    
    # Calculate primary metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report (Training Data):")
    print(classification_report(y_true, y_pred))
    
    return roc_auc

def make_predictions(model, X_test: np.ndarray) -> np.ndarray:
    """
    Generates probability predictions on the processed test data.
    
    Args:
        model: The trained model object.
        X_test: The processed test features (NumPy array).
        
    Returns:
        A NumPy array of predicted probabilities for the positive class (1.0).
    """
    print("Generating predictions on the test set...")
    # Get probability of the positive class (loan paid back = 1.0)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_proba

def save_artifacts(model, preprocessor: ColumnTransformer, score: float):
    """
    Saves the trained model and the feature preprocessor to the 'models/' directory.
    
    Args:
        model: The trained scikit-learn model.
        preprocessor: The fitted ColumnTransformer object.
        score: The primary evaluation score (ROC AUC) for metadata tracking.
    """
    # Save the model
    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to: {model_path}")

    # Save the preprocessor (CRITICAL for consistent future predictions)
    preprocessor_path = MODEL_DIR / 'feature_preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved successfully to: {preprocessor_path}")

    # Create a simple log/metadata file
    with open(MODEL_DIR / 'metadata.txt', 'w') as f:
        f.write(f"Model Type: RandomForestClassifier\n")
        f.write(f"Training ROC AUC: {score:.4f}\n")
        f.write(f"Parameters: n_estimators=200, max_depth=10, class_weight=balanced\n")
    
    print("Model metadata saved.")
