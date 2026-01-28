"""
Preprocessing pipeline (shared between binary and multi-class tasks).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class PreprocessConfig:
    target_col: str
    drop_cols: tuple[str, ...] = ("UDI", "Product ID")
    categorical_cols: tuple[str, ...] = ("Type",)
    sample_n: Optional[int] = None
    random_state: int = 42


def prepare_xy(df: pd.DataFrame, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    1) Optional sampling
    2) Drop unnecessary columns
    3) Encode categoricals
    4) Split X/y
    5) Scale X
    """
    if cfg.sample_n is not None and cfg.sample_n < len(df):
        df = df.sample(cfg.sample_n, random_state=cfg.random_state).copy()

    # Drop cols if present
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Encode categoricals
    for c in cfg.categorical_cols:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, y, scaler


def enforce_machine_failure_from_components(df: pd.DataFrame, target_col: str = "Machine failure") -> pd.DataFrame:
    """
    Optional: Ensure target is 1 if any failure component is 1 (AI4I-style datasets).
    """
    failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    if all(col in df.columns for col in failure_columns) and target_col in df.columns:
        df = df.copy()
        df[target_col] = df[failure_columns].max(axis=1)
    return df
