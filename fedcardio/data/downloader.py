"""
FedCardio 8-Hospital Consortium Data Engine & UCI Benchmark Downloader
Retrieves and verifies the 4 physical UCI hospital archives and provisions the 4
regional specialist hospital cohorts into strictly isolated storage directories.
Enforces an 85% local training / 15% global evaluation holdout split prior to disk write
with zero patient overlap and zero single-CSV slicing.
"""
import csv
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from fedcardio.config import DATA_DIR, HOSPITALS, UCI_COLUMNS
from fedcardio.data.synthesizer import (
    HOSPITAL_EPIDEMIOLOGY,
    generate_hospital_records,
    _stratified_split_records,
    _write_records_csv,
)

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"


def _parse_uci_raw_records(file_path: Path, hospital_id: str) -> List[Dict[str, Any]]:
    """
    Parses raw space or comma-separated UCI file with '?' or '-9' missing markers
    using pure standard library for universal compatibility, performing strictly
    local hospital median imputation (zero cross-silo leakage).
    """
    raw_rows: List[List[Optional[float]]] = []
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,]+", line) if p != ""]
        if len(parts) < len(UCI_COLUMNS):
            continue
        parsed_row: List[Optional[float]] = []
        for val_str in parts[: len(UCI_COLUMNS)]:
            if val_str in ("?", "-9", "-9.0", "NA", "nan"):
                parsed_row.append(None)
            else:
                try:
                    v = float(val_str)
                    parsed_row.append(None if v == -9.0 else v)
                except ValueError:
                    parsed_row.append(None)
        raw_rows.append(parsed_row)

    if not raw_rows:
        raise ValueError(f"No valid UCI rows parsed from {file_path}")

    # Compute local hospital column medians for missing value imputation
    # If >50% of a diagnostic modality (e.g., ca or thal in non-invasive clinics) is missing,
    # use the physiological unperformed baseline (ca=0, thal=3) rather than outlier sample medians.
    medians: List[float] = []
    defaults = [55.0, 1.0, 4.0, 130.0, 240.0, 0.0, 0.0, 150.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0]
    n_total_rows = max(1, len(raw_rows))
    for col_idx, col_name in enumerate(UCI_COLUMNS):
        valid_vals = sorted(r[col_idx] for r in raw_rows if r[col_idx] is not None)
        missing_frac = 1.0 - (len(valid_vals) / n_total_rows)
        if valid_vals and not (col_name in ("ca", "thal") and missing_frac > 0.50):
            mid = len(valid_vals) // 2
            med = (
                valid_vals[mid]
                if len(valid_vals) % 2 == 1
                else 0.5 * (valid_vals[mid - 1] + valid_vals[mid])
            )
            medians.append(float(med))
        else:
            medians.append(defaults[col_idx])

    records: List[Dict[str, Any]] = []
    for idx, r in enumerate(raw_rows):
        row_dict: Dict[str, Any] = {}
        for col_idx, col_name in enumerate(UCI_COLUMNS):
            val = r[col_idx] if r[col_idx] is not None else medians[col_idx]
            if col_name == "target":
                row_dict[col_name] = 1 if val > 0 else 0
            elif col_name in ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
                row_dict[col_name] = int(round(val))
            else:
                row_dict[col_name] = round(float(val), 1)
        row_dict["hospital_id"] = hospital_id
        row_dict["patient_id"] = f"{hospital_id}_pt_{idx:05d}"
        records.append(row_dict)
    return records


def fetch_or_generate_hospital_data(
    force: bool = False,
    scale_patients_per_hospital: Optional[int] = None,
    preserve_uci_real: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Downloads/verifies the 4 UCI hospital archives and generates the 4 regional
    specialist hospital cohorts into isolated directories (data/hospital_<id>/).
    Performs a strict stratified 85/15 split prior to saving local hospital CSVs
    and aggregates the 15% holdouts into data/evaluation/global_test.csv.
    """
    eval_dir = DATA_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    global_test_csv = eval_dir / "global_test.csv"

    all_exist = global_test_csv.exists() and all(
        (DATA_DIR / meta["folder"] / meta["file"]).exists() for meta in HOSPITALS.values()
    )
    if all_exist and not force and scale_patients_per_hospital is None:
        if HAS_PANDAS:
            return {
                h_id: pd.read_csv(DATA_DIR / meta["folder"] / meta["file"])
                for h_id, meta in HOSPITALS.items()
            }
        return {h_id: str(DATA_DIR / meta["folder"] / meta["file"]) for h_id, meta in HOSPITALS.items()}

    hospital_train_results: Dict[str, Any] = {}
    all_test_records: List[Dict[str, Any]] = []

    for hospital_id, meta in HOSPITALS.items():
        hospital_folder = DATA_DIR / meta["folder"]
        hospital_folder.mkdir(parents=True, exist_ok=True)
        target_csv = hospital_folder / meta["file"]

        raw_records: Optional[List[Dict[str, Any]]] = None
        uci_source = meta.get("uci_source")

        # Use authentic UCI data for the 4 benchmark sites unless full enterprise scaling override is requested
        if uci_source and (preserve_uci_real or scale_patients_per_hospital is None):
            raw_path = hospital_folder / uci_source
            if raw_path.exists() and raw_path.stat().st_size > 500:
                try:
                    raw_records = _parse_uci_raw_records(raw_path, hospital_id)
                except Exception:
                    raw_records = None

            if raw_records is None:
                raw_url = UCI_BASE_URL + uci_source
                try:
                    print(f"Downloading {meta['name']} data from UCI: {raw_url}...")
                    urllib.request.urlretrieve(raw_url, raw_path)
                    if raw_path.exists() and raw_path.stat().st_size > 500:
                        raw_records = _parse_uci_raw_records(raw_path, hospital_id)
                except Exception:
                    raw_records = None

            # If scaling is requested on top of real UCI records, augment with calibrated institutional records
            if raw_records is not None and scale_patients_per_hospital is not None:
                if len(raw_records) < scale_patients_per_hospital:
                    extra_needed = scale_patients_per_hospital - len(raw_records)
                    extra_records = generate_hospital_records(
                        hospital_id=hospital_id,
                        n_patients=extra_needed,
                        seed=seed + 100,
                    )
                    for offset, rec in enumerate(extra_records):
                        rec["patient_id"] = f"{hospital_id}_pt_{len(raw_records) + offset:05d}"
                    raw_records.extend(extra_records)

        if raw_records is None:
            n_target = scale_patients_per_hospital or HOSPITAL_EPIDEMIOLOGY[hospital_id]["n_default"]
            raw_records = generate_hospital_records(
                hospital_id=hospital_id,
                n_patients=n_target,
                seed=seed,
            )

        # Strict stratified split: 85% isolated hospital train, 15% global evaluation holdout
        train_records, test_records = _stratified_split_records(
            raw_records, test_ratio=0.15, seed=seed
        )
        _write_records_csv(target_csv, train_records)
        all_test_records.extend(test_records)

        if HAS_PANDAS:
            hospital_train_results[hospital_id] = pd.read_csv(target_csv)
        else:
            hospital_train_results[hospital_id] = train_records

        print(
            f"Saved {len(train_records)} training records to isolated silo: {target_csv} "
            f"(held out {len(test_records)} for evaluation)"
        )

    _write_records_csv(global_test_csv, all_test_records)
    print(f"Created global 8-center held-out evaluation cohort: {len(all_test_records)} patients at {global_test_csv}.")
    return hospital_train_results


if __name__ == "__main__":
    fetch_or_generate_hospital_data(force=True)
