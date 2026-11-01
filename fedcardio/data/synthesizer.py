"""
FedCardio Enterprise 8-Hospital Non-IID Clinical Cohort Synthesizer
Generates epidemiologically calibrated electronic health records (EHR) across all 8
participating global medical centers in the exact 13-concept canonical schema.
Enforces strict physical folder isolation (zero single-CSV slicing) and an 85/15
train/evaluation holdout split with zero patient record leakage.
"""
import csv
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from fedcardio.config import DATA_DIR, HOSPITALS, UCI_COLUMNS

# Epidemiological & Institutional Domain-Shift Profiles for the 8 Global Centers
HOSPITAL_EPIDEMIOLOGY: Dict[str, Dict[str, Any]] = {
    "cleveland": {
        "n_default": 600,
        "age_mean": 54.4, "age_std": 9.0,
        "male_prob": 0.68,
        "base_logit": -0.35,
        "chol_mean": 246.7, "chol_std": 48.0,
        "bp_mean": 131.6, "bp_std": 17.5,
        "fbs_prob": 0.16,
        "cath_utilization": 0.92,
        "cp_weights_healthy": [0.25, 0.35, 0.30, 0.10],
        "cp_weights_cad": [0.12, 0.13, 0.15, 0.60],
    },
    "hungary": {
        "n_default": 580,
        "age_mean": 47.9, "age_std": 8.2,
        "male_prob": 0.65,
        "base_logit": -0.75,
        "chol_mean": 238.0, "chol_std": 46.0,
        "bp_mean": 130.0, "bp_std": 16.8,
        "fbs_prob": 0.12,
        "cath_utilization": 0.08,  # Non-invasive outpatient center: rare fluoroscopy/thallium
        "cp_weights_healthy": [0.22, 0.40, 0.28, 0.10],
        "cp_weights_cad": [0.15, 0.20, 0.15, 0.50],
    },
    "zurich": {
        "n_default": 450,
        "age_mean": 55.3, "age_std": 7.8,
        "male_prob": 0.74,
        "base_logit": 0.95,        # High-acuity tertiary referral center (~74% CAD)
        "chol_mean": 222.0, "chol_std": 42.0,
        "bp_mean": 136.0, "bp_std": 19.0,
        "fbs_prob": 0.18,
        "cath_utilization": 0.65,
        "cp_weights_healthy": [0.20, 0.30, 0.30, 0.20],
        "cp_weights_cad": [0.10, 0.10, 0.12, 0.68],
    },
    "longbeach": {
        "n_default": 520,
        "age_mean": 59.4, "age_std": 8.4,
        "male_prob": 0.95,         # 95% male veterans -> Female blind spot locally!
        "base_logit": 0.85,
        "chol_mean": 236.0, "chol_std": 50.0,
        "bp_mean": 138.5, "bp_std": 20.0,
        "fbs_prob": 0.28,
        "cath_utilization": 0.45,
        "cp_weights_healthy": [0.20, 0.25, 0.35, 0.20],
        "cp_weights_cad": [0.10, 0.12, 0.15, 0.63],
    },
    "mayo": {
        "n_default": 850,
        "age_mean": 63.8, "age_std": 8.5,
        "male_prob": 0.60,
        "base_logit": 0.25,        # Geriatric multi-vessel referral center
        "chol_mean": 242.0, "chol_std": 47.0,
        "bp_mean": 137.0, "bp_std": 18.2,
        "fbs_prob": 0.22,
        "cath_utilization": 0.96,  # Full catheterization & nuclear imaging
        "cp_weights_healthy": [0.24, 0.32, 0.32, 0.12],
        "cp_weights_cad": [0.14, 0.14, 0.12, 0.60],
    },
    "hopkins": {
        "n_default": 800,
        "age_mean": 53.6, "age_std": 9.8,
        "male_prob": 0.48,         # Strong female representation (52% female!)
        "base_logit": -0.10,
        "chol_mean": 266.0, "chol_std": 54.0,
        "bp_mean": 139.0, "bp_std": 19.5,
        "fbs_prob": 0.42,          # Urban cardiometabolic & diabetic cohort
        "cath_utilization": 0.84,
        "cp_weights_healthy": [0.22, 0.38, 0.28, 0.12],
        "cp_weights_cad": [0.12, 0.26, 0.14, 0.48],  # Higher female atypical angina in CAD
    },
    "charite": {
        "n_default": 750,
        "age_mean": 50.8, "age_std": 9.4,
        "male_prob": 0.54,
        "base_logit": -0.85,       # Preventative cardiology screening (~32% CAD)
        "chol_mean": 218.0, "chol_std": 40.0,
        "bp_mean": 126.5, "bp_std": 15.5,
        "fbs_prob": 0.14,
        "cath_utilization": 0.72,
        "cp_weights_healthy": [0.28, 0.34, 0.30, 0.08],
        "cp_weights_cad": [0.15, 0.18, 0.15, 0.52],
    },
    "tokyo": {
        "n_default": 700,
        "age_mean": 57.2, "age_std": 8.8,
        "male_prob": 0.58,
        "base_logit": -0.35,       # East Asian low-lipid, hypertensive microvascular cohort
        "chol_mean": 196.0, "chol_std": 36.0,
        "bp_mean": 144.0, "bp_std": 18.8,
        "fbs_prob": 0.20,
        "cath_utilization": 0.88,
        "cp_weights_healthy": [0.25, 0.36, 0.30, 0.09],
        "cp_weights_cad": [0.14, 0.28, 0.18, 0.40],
    },
}


def _weighted_choice(rng: random.Random, options: List[int], weights: List[float]) -> int:
    total = sum(weights)
    r = rng.uniform(0.0, total)
    upto = 0.0
    for opt, w in zip(options, weights):
        if upto + w >= r:
            return opt
        upto += w
    return options[-1]


def generate_hospital_records(
    hospital_id: str,
    n_patients: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates epidemiologically calibrated clinical records for a single sovereign hospital
    in the exact 13-concept canonical schema (UCI_COLUMNS + target + hospital_id + patient_id).
    """
    spec = HOSPITAL_EPIDEMIOLOGY.get(hospital_id, HOSPITAL_EPIDEMIOLOGY["cleveland"])
    rng = random.Random(seed + sum(ord(c) * (idx + 1) for idx, c in enumerate(hospital_id)))

    records: List[Dict[str, Any]] = []
    for i in range(n_patients):
        age = round(min(max(rng.gauss(spec["age_mean"], spec["age_std"]), 26.0), 84.0))
        sex = 1 if rng.random() < spec["male_prob"] else 0

        # Underlying physiological risk latent factor (Framingham/AHA aligned)
        age_factor = (age - 54.0) * 0.055
        sex_factor = 0.35 if sex == 1 else 0.15
        metabolic_factor = 0.45 if rng.random() < spec["fbs_prob"] else 0.0
        logit = spec["base_logit"] + age_factor + sex_factor + metabolic_factor + rng.gauss(0.0, 0.65)
        cad_prob = 1.0 / (1.0 + math.exp(-logit))
        target = 1 if rng.random() < cad_prob else 0

        # Condition clinical manifestations on target and institutional profile
        if target == 1:
            cp = _weighted_choice(rng, [1, 2, 3, 4], spec["cp_weights_cad"])
            trestbps = round(min(max(rng.gauss(spec["bp_mean"] + 11.0, spec["bp_std"]), 92.0), 198.0))
            chol = round(min(max(rng.gauss(spec["chol_mean"] + 18.0, spec["chol_std"]), 125.0), 560.0))
            fbs = 1 if rng.random() < min(spec["fbs_prob"] + 0.14, 0.85) else 0
            restecg = _weighted_choice(rng, [0, 1, 2], [0.30, 0.25, 0.45])
            thalach = round(min(max(rng.gauss(134.0 - (age - 55.0) * 0.5, 19.0), 72.0), 195.0))
            exang = 1 if rng.random() < 0.58 else 0
            oldpeak = round(min(max(rng.expovariate(1.0 / 1.85), 0.1), 6.2), 1)
            slope = _weighted_choice(rng, [1, 2, 3], [0.18, 0.58, 0.24])
            if rng.random() < spec["cath_utilization"]:
                ca = _weighted_choice(rng, [0, 1, 2, 3], [0.18, 0.34, 0.30, 0.18])
                thal = _weighted_choice(rng, [3, 6, 7], [0.20, 0.22, 0.58])
            else:
                ca = 0
                thal = 3
        else:
            cp = _weighted_choice(rng, [1, 2, 3, 4], spec["cp_weights_healthy"])
            trestbps = round(min(max(rng.gauss(spec["bp_mean"] - 5.0, spec["bp_std"] * 0.85), 88.0), 180.0))
            chol = round(min(max(rng.gauss(spec["chol_mean"] - 10.0, spec["chol_std"] * 0.85), 115.0), 480.0))
            fbs = 1 if rng.random() < spec["fbs_prob"] * 0.7 else 0
            restecg = _weighted_choice(rng, [0, 1, 2], [0.68, 0.10, 0.22])
            thalach = round(min(max(rng.gauss(162.0 - (age - 50.0) * 0.4, 17.0), 85.0), 208.0))
            exang = 1 if rng.random() < 0.12 else 0
            oldpeak = round(min(max(rng.expovariate(1.0 / 0.45), 0.0), 3.2), 1)
            slope = _weighted_choice(rng, [1, 2, 3], [0.66, 0.28, 0.06])
            if rng.random() < spec["cath_utilization"]:
                ca = _weighted_choice(rng, [0, 1, 2, 3], [0.78, 0.15, 0.05, 0.02])
                thal = _weighted_choice(rng, [3, 6, 7], [0.80, 0.08, 0.12])
            else:
                ca = 0
                thal = 3

        records.append({
            "age": float(age),
            "sex": int(sex),
            "cp": int(cp),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": int(fbs),
            "restecg": int(restecg),
            "thalach": float(thalach),
            "exang": int(exang),
            "oldpeak": float(oldpeak),
            "slope": int(slope),
            "ca": int(ca),
            "thal": int(thal),
            "target": int(target),
            "hospital_id": hospital_id,
            "patient_id": f"{hospital_id}_pt_{i:05d}",
        })
    return records


def _stratified_split_records(
    records: List[Dict[str, Any]],
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic stratified 85/15 split preserving target class balance."""
    rng = random.Random(seed)
    pos = [r for r in records if int(r["target"]) == 1]
    neg = [r for r in records if int(r["target"]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_test = max(1, int(round(len(pos) * test_ratio))) if pos else 0
    n_neg_test = max(1, int(round(len(neg) * test_ratio))) if neg else 0

    test_records = pos[:n_pos_test] + neg[:n_neg_test]
    train_records = pos[n_pos_test:] + neg[n_neg_test:]
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def _write_records_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = UCI_COLUMNS + ["hospital_id", "patient_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in fieldnames})


def generate_modern_enterprise_cohort(
    n_per_hospital: int = 1200,
    seed: int = 42,
    preserve_uci_real: bool = True,
) -> Dict[str, Any]:
    """
    Generates scaled, physically isolated hospital cohorts across all 8 sovereign centers
    in the canonical 13-feature schema (UCI_COLUMNS), executing a strict 85/15 stratified
    train/holdout split prior to writing to disk.
    Zero single-CSV slicing is performed.
    """
    from fedcardio.data.downloader import fetch_or_generate_hospital_data
    return fetch_or_generate_hospital_data(
        force=True,
        scale_patients_per_hospital=n_per_hospital,
        preserve_uci_real=preserve_uci_real,
        seed=seed,
    )


# Alias for enterprise scaling CLI and API
generate_enterprise_cohort = generate_modern_enterprise_cohort


if __name__ == "__main__":
    generate_modern_enterprise_cohort(n_per_hospital=1200)
