import argparse
import os
import sys
from typing import Optional

try:
    from src.data.ingestion import (
        create_spark_session,
        extract_data_from_emr_hive,
        extract_data_from_s3,
    )
except ModuleNotFoundError:
    # Allow running this file directly without installing as a package.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.data.ingestion import (  # type: ignore
        create_spark_session,
        extract_data_from_emr_hive,
        extract_data_from_s3,
    )

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def load_data(
    source: str,
    input_filepath: Optional[str],
    hive_table: str,
    s3_path: Optional[str],
):
    """Load raw dataset from local CSV, Hive table, or S3 path as Spark DataFrame."""
    spark = create_spark_session()
    if source == "local":
        if not input_filepath:
            raise ValueError("input_filepath is required when source='local'.")
        return spark.read.csv(input_filepath, header=True, inferSchema=True)
    if source == "hive":
        return extract_data_from_emr_hive(spark, table_name=hive_table)
    if source == "s3":
        if not s3_path:
            raise ValueError("s3_path is required when source='s3'.")
        return extract_data_from_s3(spark, s3_path=s3_path)
    raise ValueError("source must be one of: local, hive, s3")


def create_derived_features(df: DataFrame) -> DataFrame:
    """Create churn label and engagement-derived features in Spark."""
    # Prioritize precise behavioral signal (recency of login) over snapshot engagement
    if "DaysSinceLastLogin" in df.columns:
        df = df.withColumn(
            "Churn_Risk",
            F.when(F.col("DaysSinceLastLogin") > F.lit(14), F.lit(1)).otherwise(F.lit(0)),
        )
    elif "EngagementLevel" in df.columns:
        # Fallback if DaysSinceLastLogin is not yet in the upstream dataset
        df = df.withColumn(
            "Churn_Risk",
            F.when(F.col("EngagementLevel") == F.lit("Low"), F.lit(1)).otherwise(F.lit(0)),
        )

    if {"SessionsPerWeek", "AvgSessionDurationMinutes"}.issubset(set(df.columns)):
        df = df.withColumn(
            "TotalWeeklyMinutes",
            F.col("SessionsPerWeek") * F.col("AvgSessionDurationMinutes"),
        )

    if {"AchievementsUnlocked", "PlayerLevel"}.issubset(set(df.columns)):
        df = df.withColumn(
            "AchievementsPerLevel",
            F.col("AchievementsUnlocked") / (F.col("PlayerLevel") + F.lit(1)),
        )

    return df


def encode_categorical_features(df: DataFrame) -> DataFrame:
    """Encode categorical columns with Spark StringIndexer."""
    import joblib
    import os
    os.makedirs("../../models", exist_ok=True)
    
    categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
    mappings = {}
    for col_name in categorical_cols:
        if col_name in df.columns:
            indexed_col = f"{col_name}_idx"
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=indexed_col,
                handleInvalid="keep",
            )
            model = indexer.fit(df)
            df = model.transform(df)
            df = df.drop(col_name).withColumnRenamed(indexed_col, col_name)
            
            # Save mapping for API
            labels = model.labels
            mappings[col_name] = {label: idx for idx, label in enumerate(labels)}
            
    joblib.dump(mappings, "../../models/category_mappings.pkl")
    return df


def drop_redundant_columns(df: DataFrame) -> DataFrame:
    """Drop non-model columns."""
    cols_to_drop = [c for c in ["PlayerID", "EngagementLevel"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(*cols_to_drop)
    return df


def save_features(df: DataFrame, output_filepath: str):
    """
    Save processed features.
    Spark writes CSV outputs as a directory of part files.
    """
    output_dir = output_filepath
    if output_filepath.lower().endswith(".csv"):
        output_dir = output_filepath.rsplit(".", 1)[0]
    df.write.mode("overwrite").option("header", True).csv(output_dir)
    print(f"Saved Spark CSV output to directory: {output_dir}")


def build_all_features(
    output_filepath: str,
    input_filepath: Optional[str] = None,
    source: str = "hive",
    hive_table: str = "game_analytics.player_churn_raw",
    s3_path: Optional[str] = None,
) -> DataFrame:
    """Run the feature pipeline with Spark-native transformations."""
    print("Loading raw data...")
    df = load_data(
        source=source,
        input_filepath=input_filepath,
        hive_table=hive_table,
        s3_path=s3_path,
    )

    print("Creating derived features...")
    df = create_derived_features(df)

    print("Encoding categoricals...")
    df = encode_categorical_features(df)

    print("Dropping redundant target columns for training dataset...")
    df = drop_redundant_columns(df)

    print(f"Saving processed features to {output_filepath}...")
    save_features(df, output_filepath)
    print("Feature engineering complete!")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build model-ready features from local CSV or EMR/S3 sources.")
    parser.add_argument("--source", choices=["local", "hive", "s3"], default="hive")
    parser.add_argument(
        "--input-path",
        default="../../data/raw/online_gaming_behavior_dataset.csv",
        help="Local CSV input path (required if source=local).",
    )
    parser.add_argument(
        "--hive-table",
        default="game_analytics.player_churn_raw",
        help="Hive table name used when source=hive.",
    )
    parser.add_argument(
        "--s3-path",
        default=None,
        help="S3 CSV path used when source=s3 (e.g. s3://bucket/raw/file.csv).",
    )
    parser.add_argument(
        "--output-path",
        default="../../data/processed/features_ready_for_modeling.csv",
        help="Processed features output path (.csv writes to directory without extension).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    build_all_features(
        output_filepath=args.output_path,
        input_filepath=args.input_path,
        source=args.source,
        hive_table=args.hive_table,
        s3_path=args.s3_path,
    )
