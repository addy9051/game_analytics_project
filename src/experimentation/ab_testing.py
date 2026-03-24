import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


def resolve_processed_data_path(filepath: str) -> Path:
    """
    Resolve processed feature path for either:
    - legacy single CSV file
    - Spark CSV output directory (part-*.csv), with optional .csv suffix in input
    """
    requested = Path(filepath)
    if requested.exists():
        return requested

    if requested.suffix.lower() == ".csv":
        spark_dir_candidate = requested.with_suffix("")
        if spark_dir_candidate.exists():
            return spark_dir_candidate

    raise FileNotFoundError(f"Processed dataset not found at {filepath}")


def read_processed_dataset(filepath: str) -> pd.DataFrame:
    """Read processed data from CSV file or Spark CSV output directory."""
    resolved_path = resolve_processed_data_path(filepath)

    if resolved_path.is_dir():
        part_files = sorted(resolved_path.glob("part-*.csv"))
        if not part_files:
            raise FileNotFoundError(f"No Spark part files found in directory: {resolved_path}")
        return pd.concat((pd.read_csv(part_file) for part_file in part_files), ignore_index=True)

    return pd.read_csv(resolved_path)

def segment_players(df: pd.DataFrame, churn_probs: np.ndarray, n_clusters=3):
    """
    Segment players using K-Means on key behavioral features and predicted churn probability.
    """
    print("Segmenting players into archetype clusters...")
    df = df.copy()
    df['Churn_Probability'] = churn_probs
    
    # Using critical engagement metrics for behavioral clustering
    features = ['PlayTimeHours', 'TotalWeeklyMinutes', 'Churn_Probability']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Segment'] = kmeans.fit_predict(df[available_features].fillna(0))
        
        # Dynamically map segments based on cluster centroids
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=available_features)
        
        # At-Risk / Churning: Highest Churn_Probability
        at_risk_idx = centroids['Churn_Probability'].idxmax()
        
        # Highly Engaged / Whales: Highest TotalWeeklyMinutes (excluding At-Risk group)
        remaining = centroids.drop(index=at_risk_idx)
        if 'TotalWeeklyMinutes' in remaining.columns:
            whales_idx = remaining['TotalWeeklyMinutes'].idxmax()
        else:
            whales_idx = remaining.index[0]
            
        casual_idx = [i for i in range(n_clusters) if i not in (at_risk_idx, whales_idx)][0]
        
        segment_map = {
            casual_idx: "Casual / Low-Risk",
            whales_idx: "Highly Engaged / Whales",
            at_risk_idx: "At-Risk / Churning"
        }
        df['Player_Persona'] = df['Segment'].map(segment_map)
    else:
        df['Player_Persona'] = "Unknown"
        
    return df

def map_interventions(df: pd.DataFrame):
    """
    Map player segments to specific tailored retention interventions.
    Incorporates Expected Value ROI tracking: (Churn_Prob * CLV) - Cost > Threshold
    """
    print("Mapping retention interventions using ROI Expected Value model...")
    offers = []
    
    # Cost & revenue assumptions
    cost_campaign_A = 5.0  # 50% off + Energy
    cost_campaign_C = 15.0 # VIP Cosmetic
    roi_threshold = 2.0
    
    for _, row in df.iterrows():
        persona = row.get('Player_Persona', 'Unknown')
        churn_prob = row.get('Churn_Probability', 0)
        
        # Estimate Lifetime Value (Proxy via InGamePurchases if available)
        if 'InGamePurchases' in df.columns:
            ltv = float(row['InGamePurchases']) * 10
        else:
            ltv = float(row.get('TotalWeeklyMinutes', 50)) * 0.1
            
        if persona == "At-Risk / Churning":
            ev = (churn_prob * ltv) - cost_campaign_A
            if ev > roi_threshold:
                offers.append("Campaign A: 50% Off Next Purchase & Free Energy Refill")
            else:
                offers.append("Campaign B: Push Notification for Weekend Bonus XP Event")
        elif persona == "Highly Engaged / Whales":
            ev = (churn_prob * ltv) - cost_campaign_C
            if ev > roi_threshold:
                offers.append("Campaign C: Exclusive VIP Cosmetic Item Unlock")
            else:
                offers.append("No Offer")
        elif persona == "Casual / Low-Risk":
            offers.append("Campaign B: Push Notification for Weekend Bonus XP Event")
        else:
            offers.append("No Offer")
            
    df['Recommended_Intervention'] = offers
    return df

def perform_ab_test_simulation(df: pd.DataFrame, target_metric='PlayTimeHours'):
    """
    Simulate an A/B test by randomly splitting 'At-Risk' players into Control and Treatment,
    and applying a simulated 15% life to the treatment group to calculate statistical significance.
    """
    print("\n--- Simulating A/B Test ---")
    at_risk = df[df['Player_Persona'] == "At-Risk / Churning"].copy()
    
    if len(at_risk) < 10:
        print("Not enough 'At-Risk' players to simulate A/B test.")
        return
        
    # Randomly assign 50% to Control and 50% to Treatment
    at_risk['Group'] = np.random.choice(["Control", "Treatment"], size=len(at_risk), p=[0.5, 0.5])
    
    # Simulate the intervention effect: 10-20% lift in the target metric for Treatment group
    if target_metric in at_risk.columns:
        treatment_mask = at_risk['Group'] == 'Treatment'
        # Add random uniform lift to represent successful intervention
        at_risk.loc[treatment_mask, target_metric] *= np.random.uniform(1.10, 1.20, size=treatment_mask.sum())
        
        control_values = at_risk[at_risk['Group'] == 'Control'][target_metric].dropna()
        treatment_values = at_risk[at_risk['Group'] == 'Treatment'][target_metric].dropna()
        
        # Execute Two-sample independent T-test
        t_stat, p_val = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        
        print(f"Control Average {target_metric}: {control_values.mean():.2f}")
        print(f"Treatment Average {target_metric}: {treatment_values.mean():.2f}")
        print(f"Result T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("--> STRONGLY STATISTICALLY SIGNIFICANT (p < 0.05): The intervention successfully retained players!")
        else:
            print("--> NOT SIGNIFICANT: The intervention did not have a reliable or distinguishable effect.")

if __name__ == "__main__":
    try:
        # Load the features previously engineered by the pipeline
        df = read_processed_dataset("../../data/processed/features_ready_for_modeling.csv")
        
        # Simulate churn probabilities that the XGBoost model would normally output real-time
        churn_probs = np.random.beta(a=2, b=5, size=len(df)) 
        
        df_segmented = segment_players(df, churn_probs)
        df_interventions = map_interventions(df_segmented)
        
        # Test the hypothesis
        perform_ab_test_simulation(df_interventions, target_metric='PlayTimeHours')
        
        # Save experimental cohort tracking for Phase 5 / BI Tools
        df_interventions.to_csv("../../data/processed/experimental_design_cohorts.csv", index=False)
        print("\nExperimental cohorts exported to 'data/processed/experimental_design_cohorts.csv' for Tableau dashboards.")
    except Exception as e:
        print("Error in experimental pipeline:", e)
