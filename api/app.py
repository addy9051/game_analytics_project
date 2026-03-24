from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import csv
from datetime import datetime
from typing import Any, Dict

app = FastAPI(title="Game Player Churn Prediction API")

# Define the expected feature schema based on our XGBoost model
class PlayerFeatures(BaseModel):
    player_id: str
    Age: float
    Gender: str
    Location: str
    GameGenre: str
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: str
    SessionsPerWeek: float
    AvgSessionDurationMinutes: float
    PlayerLevel: float
    AchievementsUnlocked: float
    TotalWeeklyMinutes: float
    AchievementsPerLevel: float

# API Key Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)):
    valid_key = os.environ["API_KEY"]  # Raises KeyError at startup if unset — intentional
    if api_key != valid_key:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

# Initialize prediction telemetry log
PREDICTION_LOG_FILE = os.getenv("PREDICTION_LOG_FILE", os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prediction_logs.csv"))
from threading import Lock
log_lock = Lock()

def log_prediction(player_id, churn_prob, risk_level, recommended_action):
    """Log prediction telemetry for feedback loop and drift detection."""
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG_FILE), exist_ok=True)
        
        with log_lock:
            file_exists = os.path.isfile(PREDICTION_LOG_FILE)
            with open(PREDICTION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "player_id", "churn_probability", "risk_tier", "recommended_action"])
                writer.writerow([datetime.utcnow().isoformat(), player_id, churn_prob, risk_level, recommended_action])
    except Exception as e:
        print(f"Failed to write prediction log: {e}")

# Load the trained XGBoost model from Phase 3
model_path = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "advanced_xgb_latest.pkl"))
try:
    xgb_model = joblib.load(model_path)
except Exception as e:
    xgb_model = None
    print(f"Warning: Model could not be loaded from {model_path}. Error: {e}")

if xgb_model is not None and hasattr(xgb_model, "feature_names_in_"):
    MODEL_FEATURES = list(xgb_model.feature_names_in_)
else:
    MODEL_FEATURES = [
        "Age",
        "Gender",
        "Location",
        "GameGenre",
        "PlayTimeHours",
        "GameDifficulty",
        "SessionsPerWeek",
        "AvgSessionDurationMinutes",
        "PlayerLevel",
        "AchievementsUnlocked",
        "TotalWeeklyMinutes",
        "AchievementsPerLevel",
    ]

# Load category mappings
mapping_path = os.getenv("MAPPINGS_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "category_mappings.pkl"))
try:
    category_mappings = joblib.load(mapping_path)
except Exception as e:
    category_mappings = {}
    print(f"Warning: Mappings could not be loaded from {mapping_path}. Error: {e}")

@app.post("/predict")
def predict_churn(features: PlayerFeatures, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    """
    Predict churn probability for a given player based on behavioral telemetry.
    """
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not loaded. Ensure Phase 3 has run.")
        
    try:
        # Convert the Pydantic basemodel directly into a pandas DataFrame mapped to the 13 training features
        data_dict: Dict[str, Any] = features.dict()
        player_id = data_dict.pop('player_id')
        
        # Apply categorical mappings
        for col in ["Gender", "Location", "GameGenre", "GameDifficulty"]:
            if col in data_dict and col in category_mappings:
                val = data_dict[col]
                data_dict[col] = category_mappings[col].get(val, len(category_mappings[col]))
                
        # Build model-aligned payload so extra request fields do not break inference.
        missing_cols = [col for col in MODEL_FEATURES if col not in data_dict]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required model features: {missing_cols}",
            )
        aligned_payload = {col: data_dict[col] for col in MODEL_FEATURES}
        df_features = pd.DataFrame([aligned_payload], columns=MODEL_FEATURES)
        
        # Predict probability of Churn (class 1)
        churn_prob = float(xgb_model.predict_proba(df_features)[0][1])
        
        # Determine archetype/action logic
        risk_level = "High" if churn_prob > 0.6 else "Medium" if churn_prob > 0.3 else "Low"
        recommended_action = "Trigger Retention Campaign" if risk_level == "High" else "Monitor"
        
        # Log payload asynchronously for feedback loop matching
        background_tasks.add_task(log_prediction, player_id, round(churn_prob, 4), risk_level, recommended_action)
        
        return {
            "player_id": player_id, 
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "recommended_action": recommended_action
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    """Monitoring endpoint for system uptime."""
    return {"status": "healthy", "model_loaded": xgb_model is not None}
