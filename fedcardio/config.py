"""
FedCardio Configuration, Clinical Metadata, and Healthcare Economics Benchmarks
Defines the 8 participating sovereign hospital nodes, clinical features, AHA/ACC physiological ranges,
federated learning hyperparameters, and actuarial ROI benchmarks.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Root Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "fedcardio" / "database"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 8 Sovereign Hospital Nodes Metadata (Physical Isolated Silos - Zero Single-CSV Slicing)
HOSPITALS: Dict[str, Dict[str, Any]] = {
    "cleveland": {
        "id": "cleveland",
        "name": "Cleveland Clinic Foundation",
        "city": "Cleveland, OH",
        "country": "United States",
        "region": "North America",
        "lat": 41.5034,
        "lon": -81.6212,
        "folder": "hospital_cleveland",
        "file": "cleveland_patients.csv",
        "uci_source": "processed.cleveland.data",
        "tier": "UCI Benchmark + Academic Cath Lab",
        "specialty": "Interventional Cardiology & Fluoroscopy",
        "blind_spot": "Western tertiary referral bias; under-represents East Asian low-lipid vascular profiles.",
        "annual_admissions": 4200,
        "description": "Tertiary academic medical center with specialized catheterization and nuclear stress labs."
    },
    "hungary": {
        "id": "hungary",
        "name": "Hungarian Institute of Cardiology",
        "city": "Budapest",
        "country": "Hungary",
        "region": "Central Europe",
        "lat": 47.4979,
        "lon": 19.0402,
        "folder": "hospital_hungary",
        "file": "hungarian_patients.csv",
        "uci_source": "processed.hungarian.data",
        "tier": "UCI Benchmark + National Registry",
        "specialty": "Non-Invasive Outpatient Triage",
        "blind_spot": "Lacks routine fluoroscopy (ca) & thallium (thal) scans; misses silent multi-vessel ischemia.",
        "annual_admissions": 3800,
        "description": "National cardiology center serving Central and Eastern European outpatient cohorts."
    },
    "zurich": {
        "id": "zurich",
        "name": "University Hospital Zurich & Basel",
        "city": "Zurich",
        "country": "Switzerland",
        "region": "Western Europe",
        "lat": 47.3769,
        "lon": 8.5417,
        "folder": "hospital_zurich",
        "file": "switzerland_patients.csv",
        "uci_source": "processed.switzerland.data",
        "tier": "UCI Benchmark + Acute Referral",
        "specialty": "Refractory Acute Coronary Syndrome",
        "blind_spot": "76% high-acuity disease prevalence causes severe false-positive catheterizations on low-risk walk-ins.",
        "annual_admissions": 2900,
        "description": "Leading European tertiary referral hospital with high-acuity ischemic cohorts."
    },
    "longbeach": {
        "id": "longbeach",
        "name": "V.A. Medical Center Long Beach",
        "city": "Long Beach, CA",
        "country": "United States",
        "region": "North America",
        "lat": 33.7701,
        "lon": -118.1937,
        "folder": "hospital_longbeach",
        "file": "va_patients.csv",
        "uci_source": "processed.va.data",
        "tier": "UCI Benchmark + Veterans Health",
        "specialty": "Geriatric Veteran Comorbidity Care",
        "blind_spot": "95% male veteran cohort creates a critical diagnostic blind spot for female microvascular angina.",
        "annual_admissions": 3100,
        "description": "Veterans Affairs medical center with elderly male high-comorbidity cardiovascular profiles."
    },
    "mayo": {
        "id": "mayo",
        "name": "Mayo Clinic Heart Institute",
        "city": "Rochester, MN",
        "country": "United States",
        "region": "North America",
        "lat": 44.0225,
        "lon": -92.4668,
        "folder": "hospital_mayo",
        "file": "mayo_patients.csv",
        "uci_source": None,
        "tier": "Enterprise Specialist Node",
        "specialty": "Complex Multi-Vessel & Geriatric Surgery",
        "blind_spot": "Skewed toward elderly multi-vessel surgical referrals; under-calibrated for young ambulatory screening.",
        "annual_admissions": 5400,
        "description": "High-volume surgical and catheterization referral center specializing in multi-vessel coronary disease."
    },
    "hopkins": {
        "id": "hopkins",
        "name": "The Johns Hopkins Hospital",
        "city": "Baltimore, MD",
        "country": "United States",
        "region": "North America",
        "lat": 39.2964,
        "lon": -76.5926,
        "folder": "hospital_hopkins",
        "file": "hopkins_patients.csv",
        "uci_source": None,
        "tier": "Enterprise Specialist Node",
        "specialty": "Urban Cardiometabolic & Diabetic Care",
        "blind_spot": "High diabetes (fbs) and hyperlipidemia prevalence; under-weights normoglycemic exercise ischemia.",
        "annual_admissions": 4900,
        "description": "Academic urban medical center with high representation of female and cardiometabolic syndrome patients."
    },
    "charite": {
        "id": "charite",
        "name": "Charite - Universitatsmedizin Berlin",
        "city": "Berlin",
        "country": "Germany",
        "region": "Western Europe",
        "lat": 52.5256,
        "lon": 13.3775,
        "folder": "hospital_charite",
        "file": "charite_patients.csv",
        "uci_source": None,
        "tier": "Enterprise Specialist Node",
        "specialty": "Preventative Cardiology & Early Screening",
        "blind_spot": "Low baseline event rate (32%) and mild symptom distribution; under-calls severe calcified 3-vessel CAD.",
        "annual_admissions": 4600,
        "description": "Europe's largest university hospital driving early asymptomatic and preventative cardiac screening."
    },
    "tokyo": {
        "id": "tokyo",
        "name": "Tokyo Cardiovascular Institute",
        "city": "Tokyo",
        "country": "Japan",
        "region": "East Asia",
        "lat": 35.6762,
        "lon": 139.6503,
        "folder": "hospital_tokyo",
        "file": "tokyo_patients.csv",
        "uci_source": None,
        "tier": "Enterprise Specialist Node",
        "specialty": "East Asian Hypertensive & Vasospastic CAD",
        "blind_spot": "Lower baseline serum cholesterol (mean 198 mg/dL) with higher atypical/vasospastic angina presentation.",
        "annual_admissions": 4100,
        "description": "Premier East Asian cardiac center capturing low-BMI hypertensive and microvascular angina phenotypes."
    }
}

# Core 4 Historical UCI Nodes vs Full 8-Hospital Enterprise Consortium
UCI_BENCHMARK_HOSPITALS: List[str] = ["cleveland", "hungary", "zurich", "longbeach"]
ENTERPRISE_SPECIALIST_HOSPITALS: List[str] = ["mayo", "hopkins", "charite", "tokyo"]

# Real-World Canonical Clinical Feature Names (13 predictors + 1 target)
UCI_COLUMNS = [
    "age",        # Age in years
    "sex",        # 1 = male; 0 = female
    "cp",         # Chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic
    "trestbps",   # Resting blood pressure (mm Hg on admission to hospital)
    "chol",       # Serum cholesterol in mg/dl
    "fbs",        # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",    # Resting electrocardiographic results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)
    "thalach",    # Maximum heart rate achieved during exercise
    "exang",      # Exercise-induced angina (1 = yes; 0 = no)
    "oldpeak",    # ST depression induced by exercise relative to rest
    "slope",      # Slope of the peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
    "ca",         # Number of major vessels (0-3) colored by fluoroscopy
    "thal",       # Thallium scan: 3=normal, 6=fixed defect, 7=reversible defect
    "target"      # Heart disease diagnosis: 0 = <50% diameter narrowing, 1 = >50% narrowing
]

# Continuous vs Categorical Clinical Features for scaling
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# AHA/ACC Standard Clinical Ranges and Feature Definitions
CONTINUOUS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (25.0, 85.0),
    "trestbps": (80.0, 200.0),
    "chol": (100.0, 600.0),
    "thalach": (60.0, 220.0),
    "oldpeak": (0.0, 6.5),
    "ca": (0.0, 3.0),
}

CATEGORICAL_CATEGORIES: Dict[str, List[int]] = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

BINARY_FEATURES: List[str] = ["sex", "fbs", "exang"]

TOTAL_FEATURES: int = 22

# Canonical 13 clinical concepts (in order of UCI feature columns)
CLINICAL_CONCEPTS: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Canonical 22-dimensional encoded feature column names
ENCODED_FEATURE_NAMES: List[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
    "sex", "fbs", "exang",
    "cp_1", "cp_2", "cp_3", "cp_4",
    "restecg_0", "restecg_1", "restecg_2",
    "slope_1", "slope_2", "slope_3",
    "thal_3", "thal_6", "thal_7"
]

# Clinical Physiological Bounds and Units for Clinician Validation
CLINICAL_BOUNDS: Dict[str, Dict[str, Any]] = {
    "age": {"label": "Age", "unit": "years", "min": 25, "max": 85, "default": 55},
    "sex": {"label": "Biological Sex", "options": ["Female", "Male"], "default": "Male"},
    "cp": {"label": "Chest Pain Type", "options": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], "default": "Typical Angina"},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "min": 80, "max": 200, "default": 130},
    "chol": {"label": "Serum Cholesterol", "unit": "mg/dL", "min": 100, "max": 600, "default": 240},
    "fbs": {"label": "Fasting Blood Sugar > 120 mg/dL", "options": ["Normal (<= 120)", "Elevated (> 120)"], "default": "Normal (<= 120)"},
    "restecg": {"label": "Resting ECG", "options": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], "default": "Normal"},
    "thalach": {"label": "Max Heart Rate Achieved", "unit": "bpm", "min": 60, "max": 220, "default": 150},
    "exang": {"label": "Exercise Induced Angina", "options": ["No", "Yes"], "default": "No"},
    "oldpeak": {"label": "ST Depression (Oldpeak)", "unit": "mm", "min": 0.0, "max": 6.5, "default": 1.0},
    "slope": {"label": "ST Slope", "options": ["Upsloping", "Flat", "Downsloping"], "default": "Flat"},
    "ca": {"label": "Major Vessels Colored (Fluoroscopy)", "unit": "vessels (0-3)", "min": 0, "max": 3, "default": 0},
    "thal": {"label": "Thallium Stress Test", "options": ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"], "default": "Normal (3)"}
}

# Healthcare Economics & Actuarial Benchmarks (AHA / CMS / IBM Ponemon Healthcare)
ACTUARIAL_BENCHMARKS: Dict[str, float] = {
    "cost_prevented_ami_usd": 42500.0,           # Emergency AMI ICU admission + emergent PCI + heart failure care
    "cost_avoided_angiography_usd": 12400.0,     # Unnecessary invasive coronary catheterization + complication risk
    "preventative_care_cost_usd": 1850.0,        # Outpatient stress echo + high-intensity statin + follow-up
    "cms_hrrp_penalty_saved_per_site_usd": 185000.0,  # CMS 30-day readmission penalty mitigation per hospital
    "hipaa_breach_cost_per_record_usd": 499.0,   # IBM Security / Ponemon Healthcare breach cost per patient record
    "annual_platform_subscription_per_site_usd": 45000.0,
}

@dataclass
class FLConfig:
    num_rounds: int = 5
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    optimizer: str = "SGD"
    momentum: float = 0.9
    strategy: str = "FedProx"       # 'FedAvg', 'FedProx', 'FedMedian', or 'TrimmedMean'
    mu: float = 0.01               # FedProx proximal regularization strength
    trim_ratio: float = 0.2        # Trimming fraction beta for TrimmedMean
    byzantine_nodes: int = 0       # Number of simulated compromised/poisoned hospital nodes
    use_dp: bool = True
    epsilon: float = 3.0            # Target privacy budget
    delta: float = 1e-4             # Target delta
    max_grad_norm: float = 1.0      # Per-sample gradient clipping threshold
    device: str = "cpu"
    seed: int = 42

DEFAULT_FL_CONFIG = FLConfig()
