from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Game Player Churn Prediction API")

# Define the expected feature schema based on our XGBoost model
class PlayerFeatures(BaseModel):
    player_id: str
    Age: float
    Gender: int
    Location: int
    GameGenre: int
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: int
    SessionsPerWeek: float
    AvgSessionDurationMinutes: float
    PlayerLevel: float
    AchievementsUnlocked: float
    TotalWeeklyMinutes: float
    AchievementsPerLevel: float

# Load the trained XGBoost model from Phase 3
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "advanced_xgb.pkl")
try:
    xgb_model = joblib.load(model_path)
except Exception as e:
    xgb_model = None
    print(f"Warning: Model could not be loaded from {model_path}. Error: {e}")

@app.post("/predict")
def predict_churn(features: PlayerFeatures):
    """
    Predict churn probability for a given player based on behavioral telemetry.
    """
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not loaded. Ensure Phase 3 has run.")
        
    try:
        # Convert the Pydantic basemodel directly into a pandas DataFrame mapped to the 13 training features
        data_dict = features.dict()
        player_id = data_dict.pop('player_id')
        
        # Order must match training exactly (the Dataframe ensures column names map properly in xgboost)
        df_features = pd.DataFrame([data_dict])
        
        # Predict probability of Churn (class 1)
        churn_prob = float(xgb_model.predict_proba(df_features)[0][1])
        
        # Determine archetype/action logic
        risk_level = "High" if churn_prob > 0.6 else "Medium" if churn_prob > 0.3 else "Low"
        
        return {
            "player_id": player_id, 
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "recommended_action": "Trigger Retention Campaign" if risk_level == "High" else "Monitor"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    """Monitoring endpoint for system uptime."""
    return {"status": "healthy", "model_loaded": xgb_model is not None}
