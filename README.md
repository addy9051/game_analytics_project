# Player Churn Prediction & Engagement Optimization Platform

An end-to-end analytics platform that predicts player churn, identifies at-risk segments, and triggers targeted interventions to improve retention and monetization.

## Project Structure
- `src/` - Source code for ingestion, features, models and experimentation
- `notebooks/` - Jupyter notebooks for Exploratory Data Analysis
- `api/` - Deployment code for the Fast/FastAPI server

## Recent Improvements

### 🐘 Big Data Scale Integration
- **PySpark Migration**: Transitioned feature engineering and data ingestion pipelines from Pandas to PySpark, unlocking distributed processing over scale-out S3/EMR datasets.

### 🔧 Code Cleanup & Streamlining
- **Removed Mismatched Preprocessing**: Eliminated Z-score scaling from the Spark pipeline to prevent training-serving skew, integrating `StandardScaler` directly into a Sklearn `Pipeline`.
- **Simplified Architectures**: Replaced the faux-sequential LSTM network with a proper Dense Feedforward Neural Network, suitable for tabular/cross-sectional player data. Removed empty A/B test stub functions.

### 🚀 Production API Readiness
- **Robust Model Validation**: Extract Spark `StringIndexer` mappings into `category_mappings.pkl` for accurate end-to-end categorical encoding at inference time.
- **Model Versioning**: Enforced timestamped joblib model artifacts alongside a dynamic `advanced_xgb_latest.pkl` alias.
- **Docker Portability**: The `/predict` API dynamically resolves models via `MODEL_PATH` and `MAPPINGS_PATH` environment variables for S3/GCS compatibility at startup.
- **Secured Endpoints**: Interlocked the FastAPI POST `/predict` route behind `X-API-Key` authentication headers.

### 📈 Business Impact & Modeling Upgrades
- **Actionable Churn Definition**: Redefined target labels towards a predictive `DaysSinceLastLogin > 14` threshold rather than point-in-time snapshots.
- **ROI-Driven Interventions**: Built an Expected Value formula into A/B testing clustering `(Churn_Prob × LTV) - Campaign_Cost > ROI_Threshold`, ensuring retention budgets focus on profitable "Whales".
- **Revenue-Weighted AUC Metrics**: Adjusted `evaluate_model.py` to heavily weight False Negatives towards players with high `InGamePurchases`, directly aligning model optimization with actual business KPIs.
- **Telemetry Feedback Loops**: Added asynchronous async prediction logging into `prediction_logs.csv` tracing player risks to enable downstream drift monitoring and lift analyses.

## Setup
1. Create a virtual environment (`python -m venv venv`)
2. Install requirements (`pip install -r requirements.txt`)
