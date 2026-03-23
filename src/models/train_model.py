import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
import joblib
import os

def load_and_split_data(filepath: str):
    """Load the processed dataset and split into train/test."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Churn_Risk'])
    y = df['Churn_Risk']
    # 80/20 train-test split stratified on the target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_baseline_model(X_train, y_train):
    """Train interpretable baseline model (logistic regression)."""
    print("Training Baseline Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
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

def build_lstm_model(input_shape):
    """Build an LSTM-based deep learning model for POC."""
    print("Building LSTM Model...")
    model = Sequential()
    # Reshape input to (samples, time_steps, features). 
    # For a cross-sectional dataset, we treat all features as 1 time step.
    model.add(Reshape((1, input_shape), input_shape=(input_shape,)))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    filepath = "../../data/processed/features_ready_for_modeling.csv"
    if os.path.exists(filepath):
        print("Loading processed data...")
        X_train, X_test, y_train, y_test = load_and_split_data(filepath)
        
        # Train Models
        baseline_lr = train_baseline_model(X_train, y_train)
        advanced_xgb = train_advanced_model(X_train, y_train)
        
        print("\nTraining Deep Learning Architecture...")
        lstm_model = build_lstm_model(X_train.shape[1])
        lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
        
        # Save models for deployment
        os.makedirs("../../models", exist_ok=True)
        joblib.dump(baseline_lr, "../../models/baseline_lr.pkl")
        joblib.dump(advanced_xgb, "../../models/advanced_xgb.pkl")
        lstm_model.save("../../models/lstm_model.h5")
        
        print("Models successfully trained and saved!")
    else:
        print(f"Data file not found: {filepath}. Run build_features.py first.")
