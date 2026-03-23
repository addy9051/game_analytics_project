import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw game behavior dataset."""
    return pd.read_csv(filepath)

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on existing ones.
    Since we don't have explicit time-series logs, we use the aggregates provided.
    """
    df = df.copy()
    
    # Target definition: Convert EngagementLevel 'Low' to churn prediction target (1=Churn/Low, 0=Retain/High-Med)
    if 'EngagementLevel' in df.columns:
        df['Churn_Risk'] = df['EngagementLevel'].apply(lambda x: 1 if x == 'Low' else 0)
    
    # Feature Engineering: Weekly engagement intensity
    df['TotalWeeklyMinutes'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
    
    # Feature Engineering: Achievement efficiency
    # Avoid division by zero
    df['AchievementsPerLevel'] = df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns for modeling."""
    df = df.copy()
    categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features using StandardScaler."""
    df = df.copy()
    num_cols = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 
                'PlayerLevel', 'AchievementsUnlocked', 'TotalWeeklyMinutes', 'AchievementsPerLevel']
    
    scaler = StandardScaler()
    # Only scale columns that exist
    cols_to_scale = [c for c in num_cols if c in df.columns]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df

def build_all_features(input_filepath: str, output_filepath: str):
    """Run the entire feature engineering pipeline and save the result."""
    print("Loading raw data...")
    df = load_data(input_filepath)
    
    print("Creating derived features...")
    df = create_derived_features(df)
    
    print("Encoding categoricals...")
    df = encode_categorical_features(df)
    
    print("Scaling numericals...")
    df = scale_numerical_features(df)
    
    print("Dropping redundant target columns for training dataset...")
    # Drop original ID and EngagementLevel to avoid target leakage since Churn_Risk is derived from it
    if 'PlayerID' in df.columns:
        df = df.drop(columns=['PlayerID'])
    if 'EngagementLevel' in df.columns:
        df = df.drop(columns=['EngagementLevel'])
        
    print(f"Saving processed features to {output_filepath}...")
    df.to_csv(output_filepath, index=False)
    print("Feature engineering complete!")
    return df

if __name__ == "__main__":
    # Example local usage
    input_path = "../../data/raw/online_gaming_behavior_dataset.csv"
    output_path = "../../data/processed/features_ready_for_modeling.csv"
    
    # Run locally
    import os
    os.makedirs("../../data/processed", exist_ok=True)
    build_all_features(input_path, output_path)
