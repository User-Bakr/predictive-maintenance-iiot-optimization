"""
Data loading utilities for ACO predictive maintenance experiments.

Note: Datasets are not included in the repo.
"""

from __future__ import annotations
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV dataset from a local path."""
    df = pd.read_csv(csv_path)
    return df


def basic_sanity_checks(df: pd.DataFrame, id_cols: list[str] | None = None) -> None:
    """Print simple sanity checks (schema, duplicates)."""
    print("Features non-null values and data type:")
    df.info()

    if id_cols:
        for col in id_cols:
            if col in df.columns:
                n = df.shape[0]
                unique_n = df[col].nunique(dropna=False)
                print(f"Check for duplicates in '{col}':", unique_n != n)
