import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate using AUC and classification reports."""
    print(f"\n--- Evaluating {model_name} ---")
    
    # Differentiate between Sklearn/XGBoost (predict_proba) and Keras (predict)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test).ravel()
        
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print(classification_report(y_test, y_pred))
    
    # Standard AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Revenue-Weighted AUC (Prioritizes False Negatives on Whales)
    if 'InGamePurchases' in X_test.columns:
        # Avoid zero weights for non-spenders so they aren't completely ignored
        weights = X_test['InGamePurchases'].values + 0.1
        weighted_roc_auc = roc_auc_score(y_test, y_pred_proba, sample_weight=weights)
        print(f"Standard ROC AUC Score: {roc_auc:.4f}")
        print(f"Revenue-Weighted ROC AUC Score: {weighted_roc_auc:.4f} (Optimized for LTV)")
        roc_auc_to_return = weighted_roc_auc
    else:
        print(f"Standard ROC AUC Score: {roc_auc:.4f}")
        roc_auc_to_return = roc_auc
        
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    return roc_auc_to_return

def generate_shap_values(model, X_test, output_path="shap_summary.png"):
    """Apply SHAP values to identify key drivers of churn."""
    print(f"\nGenerating SHAP values (TreeExplainer) to {output_path}...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Generate summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP Feature Importance (Drivers of Churn)')
        plt.tight_layout()
        plt.savefig(output_path)
        print("SHAP plot successfully saved!")
    except Exception as e:
        print(f"Error generating SHAP: {e}")

if __name__ == "__main__":
    # Local execution for evaluation
    try:
        try:
            from train_model import load_and_split_data
        except ModuleNotFoundError:
            from src.models.train_model import load_and_split_data

        filepath = "../../data/processed/features_ready_for_modeling.csv"
        X_train, X_test, y_train, y_test = load_and_split_data(filepath)
        
        print("Loading trained XGBoost model for evaluation...")
        xgb_model = joblib.load("../../models/advanced_xgb.pkl")
        
        evaluate_model(xgb_model, X_test, y_test, "XGBoost Classifier")
        generate_shap_values(xgb_model, X_test, output_path="../../models/shap_summary.png")
    except Exception as e:
        print(f"Check if models have been trained and data is processed. Error: {e}")
