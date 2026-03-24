import pandas as pd
from pathlib import Path

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
