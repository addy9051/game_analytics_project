import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.data_io import read_processed_dataset


def load_and_split_data(filepath: str):
    """Load the processed dataset and split into train/test."""
    df = read_processed_dataset(filepath)
    X = df.drop(columns=['Churn_Risk'])
    y = df['Churn_Risk']
    # 80/20 train-test split stratified on the target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_baseline_model(X_train, y_train):
    """Train interpretable baseline model (logistic regression)."""
    print("Training Baseline Logistic Regression...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model

def train_advanced_model(X_train, y_train):
    """Implement advanced gradient boosting model (XGBoost)."""
    print("Training Advanced XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def build_dense_model(input_shape):
    """Build a simple Dense network for POC."""
    print("Building Dense Model...")
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    filepath = "../../data/processed/features_ready_for_modeling.csv"
    try:
        print("Loading processed data...")
        X_train, X_test, y_train, y_test = load_and_split_data(filepath)
        
        # Hold back LTV target so it is strictly used as an evaluation weight
        if 'InGamePurchases' in X_train.columns:
            X_train_features = X_train.drop(columns=['InGamePurchases'])
        else:
            X_train_features = X_train
            
        # Train Models
        baseline_lr = train_baseline_model(X_train_features, y_train)
        advanced_xgb = train_advanced_model(X_train_features, y_train)
        
        print("\nTraining Deep Learning Architecture...")
        dl_model = build_dense_model(X_train_features.shape[1])
        dl_model.fit(X_train_features, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
        
        # Save models for deployment
        import shutil
        from datetime import datetime
        os.makedirs("../../models", exist_ok=True)
        joblib.dump(baseline_lr, "../../models/baseline_lr.pkl")
        
        # Timestamp based model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xgb_filename = f"advanced_xgb_{timestamp}.pkl"
        xgb_path = os.path.join("../../models", xgb_filename)
        joblib.dump(advanced_xgb, xgb_path)
        
        # Create latest symlink or copy
        latest_path = "../../models/advanced_xgb_latest.pkl"
        try:
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(xgb_filename, latest_path)
        except OSError:
            # Fallback for Windows if symlink privilege is missing
            shutil.copy(xgb_path, latest_path)
            
        dl_model.save("../../models/dense_model.h5")
        
        print("Models successfully trained and saved!")
    except FileNotFoundError:
        print(f"Data file not found: {filepath}. Run build_features.py first.")
