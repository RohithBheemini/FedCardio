"""
FedCardio Isolated PyTorch Clinical DataLoaders
Enforces strict client isolation: loaders only read from their assigned hospital folder.
Applies universal physiological min-max normalization and categorical one-hot encoding
to yield consistent, mathematically stable 22-dimensional feature tensors across all nodes.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from fedcardio.config import (
    DATA_DIR,
    HOSPITALS,
    CONTINUOUS_BOUNDS,
    CATEGORICAL_CATEGORIES,
    BINARY_FEATURES,
    TOTAL_FEATURES,
    ENCODED_FEATURE_NAMES,
)


def encode_clinical_features(
    data: Union[pd.DataFrame, Dict[str, Any], pd.Series]
) -> np.ndarray:
    """
    Transforms raw clinical features into a normalized 22-dimensional feature tensor.
    Applies universal physiological min-max normalization for continuous vitals,
    binary clipping to [0.0, 1.0], and one-hot encoding for categorical variables.

    Parameters:
        data: A pandas DataFrame, dictionary, or Series with raw clinical features.

    Returns:
        np.ndarray of shape (N, 22) and dtype float32 with values bounded in [0.0, 1.0].
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data.to_dict()])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type for clinical feature encoding: {type(data)}")

    n_samples = len(df)
    encoded_columns: List[np.ndarray] = []

    # 1. Universal Physiological Min-Max Normalization for Continuous Variables (6 features)
    for col, (f_min, f_max) in CONTINUOUS_BOUNDS.items():
        if col not in df.columns:
            raise KeyError(f"Missing required clinical continuous feature: '{col}'")
        raw_vals = df[col].to_numpy(dtype=np.float32)
        norm_vals = np.clip((raw_vals - f_min) / (f_max - f_min), 0.0, 1.0)
        encoded_columns.append(norm_vals.reshape(-1, 1))

    # 2. Binary Clinical Indicators (3 features)
    for col in BINARY_FEATURES:
        if col not in df.columns:
            raise KeyError(f"Missing required clinical binary feature: '{col}'")
        raw_vals = df[col].to_numpy(dtype=np.float32)
        bin_vals = np.clip(raw_vals, 0.0, 1.0)
        encoded_columns.append(bin_vals.reshape(-1, 1))

    # 3. Categorical Variables One-Hot Encoding (13 features)
    for col, categories in CATEGORICAL_CATEGORIES.items():
        if col not in df.columns:
            raise KeyError(f"Missing required clinical categorical feature: '{col}'")
        raw_vals = df[col].to_numpy()
        for cat in categories:
            one_hot = (raw_vals == cat).astype(np.float32)
            encoded_columns.append(one_hot.reshape(-1, 1))

    feature_matrix = np.hstack(encoded_columns).astype(np.float32)
    assert feature_matrix.shape == (n_samples, TOTAL_FEATURES), (
        f"Encoded feature shape mismatch: expected ({n_samples}, {TOTAL_FEATURES}), "
        f"got {feature_matrix.shape}"
    )
    return feature_matrix


class IsolatedHospitalDataset:
    """
    Manages an isolated hospital's local clinical data.
    Uses universal physiological normalization to ensure zero cross-silo contamination
    while guaranteeing mathematical consistency across sovereign sites.
    """
    def __init__(
        self,
        hospital_id: str,
        batch_size: int = 16,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.hospital_id = hospital_id
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed

        meta = HOSPITALS.get(hospital_id)
        if not meta:
            raise ValueError(
                f"Unknown hospital_id: '{hospital_id}'. Expected one of {list(HOSPITALS.keys())}"
            )

        self.data_path = DATA_DIR / meta["folder"] / meta["file"]
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Missing isolated hospital data file: {self.data_path}. Run downloader first."
            )

        self._load_and_prepare()

    def _load_and_prepare(self) -> None:
        # Strict local file read from physical silo
        df = pd.read_csv(self.data_path)

        # Local train/val split within this hospital
        train_df, val_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df["target"] if len(df["target"].unique()) > 1 else None,
        )

        X_train = encode_clinical_features(train_df)
        y_train = train_df["target"].to_numpy(dtype=np.float32)

        X_val = encode_clinical_features(val_df)
        y_val = val_df["target"].to_numpy(dtype=np.float32)

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).unsqueeze(1),
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val).unsqueeze(1),
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.n_samples = len(X_train)


def get_hospital_loaders(
    hospital_id: str, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, int]:
    """Helper to return train_loader, val_loader, and n_samples for a given hospital."""
    iso = IsolatedHospitalDataset(hospital_id, batch_size=batch_size)
    return iso.train_loader, iso.val_loader, iso.n_samples


def get_global_evaluation_loader(
    batch_size: int = 32
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Loads the multi-center held-out evaluation set for consortium evaluation.
    Applies identical universal physiological normalization with zero test data fitting.
    """
    eval_csv = DATA_DIR / "evaluation" / "global_test.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_csv}")

    df = pd.read_csv(eval_csv)
    X = encode_clinical_features(df)
    y = df["target"].to_numpy(dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, df
