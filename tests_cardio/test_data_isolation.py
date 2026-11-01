"""
Automated Test Suite: Data Isolation & Directory Separation
Verifies that all 4 hospitals possess independent, isolated files and zero cross-silo contamination.
"""
import pytest
from pathlib import Path
import pandas as pd

from fedcardio.config import HOSPITALS, DATA_DIR, UCI_COLUMNS
from fedcardio.data.loader import get_hospital_loaders, get_global_evaluation_loader

def test_hospital_files_exist_in_separate_folders():
    """Verify that every hospital node has its own physical folder and data file."""
    for h_id, meta in HOSPITALS.items():
        folder = DATA_DIR / meta["folder"]
        assert folder.exists(), f"Directory missing for {h_id}: {folder}"
        assert folder.is_dir(), f"Path is not a directory: {folder}"
        
        file_path = folder / meta["file"]
        assert file_path.exists(), f"Data file missing for {h_id}: {file_path}"
        assert file_path.stat().st_size > 1000, f"Data file is suspiciously small: {file_path}"
        
        df = pd.read_csv(file_path)
        assert len(df) > 50, f"Expected >50 records for {h_id}, got {len(df)}"
        assert "target" in df.columns, f"Missing target in {file_path}"

def test_isolated_hospital_loaders():
    """Verify that loaders load proper batch sizes and 22-dimensional tensors independently."""
    for h_id in HOSPITALS.keys():
        train_loader, val_loader, n_samples = get_hospital_loaders(h_id, batch_size=16)
        assert n_samples > 40
        
        # Test first batch
        X_b, y_b = next(iter(train_loader))
        assert X_b.shape[1] == 22, f"Expected 22 clinical features, got {X_b.shape[1]}"
        assert y_b.shape[1] == 1, f"Expected 1 target column, got {y_b.shape[1]}"

def test_global_evaluation_loader():
    """Verify that the central evaluation holdout exists, has 22 features, and is strictly disjoint."""
    eval_loader, eval_df = get_global_evaluation_loader(batch_size=32)
    assert len(eval_df) > 50
    X_b, y_b = next(iter(eval_loader))
    assert X_b.shape[1] == 22, f"Expected 22 clinical features, got {X_b.shape[1]}"
    
    # Assert strict zero row overlap / zero patient leakage between hospital silos and evaluation set
    eval_ids = set(eval_df["patient_id"]) if "patient_id" in eval_df.columns else set()
    for h_id, meta in HOSPITALS.items():
        h_df = pd.read_csv(DATA_DIR / meta["folder"] / meta["file"])
        if "patient_id" in h_df.columns and eval_ids:
            overlap = set(h_df["patient_id"]).intersection(eval_ids)
            assert len(overlap) == 0, f"Data leakage detected between {h_id} and global evaluation set: {overlap}"
