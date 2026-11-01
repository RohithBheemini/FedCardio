"""
FedCardio Clinical Explainability (XAI)
Implements Grouped Permutation Coalitional SHAP across the 13 clinical concepts,
where one-hot categorical feature sets (e.g., cp_1..4, restecg_0..2, slope_1..3, thal_3..7)
are toggled cohesively as single clinical entities against a physiological median background vector.
Mathematically guarantees adherence to the Shapley Efficiency Axiom: sum(attributions) == f(x) - f(x_bg).
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.config import (
    CHECKPOINT_DIR,
    TOTAL_FEATURES,
    CLINICAL_CONCEPTS,
    ENCODED_FEATURE_NAMES,
)
from fedcardio.data.loader import get_global_evaluation_loader, encode_clinical_features

FRIENDLY_FEATURE_NAMES = {
    "age": "Age",
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Slope",
    "ca": "Major Vessels Colored",
    "thal": "Thallium Stress Test",
}

# Grouped index mapping of 22 encoded features into 13 clinical concepts
CONCEPT_INDICES: Dict[str, List[int]] = {
    "age": [0],
    "trestbps": [1],
    "chol": [2],
    "thalach": [3],
    "oldpeak": [4],
    "ca": [5],
    "sex": [6],
    "fbs": [7],
    "exang": [8],
    "cp": [9, 10, 11, 12],
    "restecg": [13, 14, 15],
    "slope": [16, 17, 18],
    "thal": [19, 20, 21],
}


class ClinicalSHAPExplainer:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = ClinicalNet(in_features=TOTAL_FEATURES)
        ckpt = model_path or (CHECKPOINT_DIR / "best_model.pt")
        if ckpt.exists():
            try:
                self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            except Exception:
                pass
        self.model.eval()

        # Load reference evaluation cohort and derive physiological median background vector (22 features)
        eval_loader, eval_df = get_global_evaluation_loader()
        X_eval = encode_clinical_features(eval_df)
        self.bg_vector = np.median(X_eval, axis=0).astype(np.float32)

        # Ensure one-hot categorical sub-vectors in the background represent valid single choices
        for cat_name in ["cp", "restecg", "slope", "thal"]:
            cols = CONCEPT_INDICES[cat_name]
            best_cat = cols[int(np.argmax(self.bg_vector[cols]))]
            for c in cols:
                self.bg_vector[c] = 1.0 if c == best_cat else 0.0

        self.clinical_concepts = CLINICAL_CONCEPTS

    def explain_patient(
        self,
        raw_patient_dict: Dict[str, Any],
        n_permutations: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Computes Grouped Permutation Coalitional SHAP feature attributions
        across the 13 clinical concepts. Categorical groups are toggled as coalitions.
        Guarantees that the sum of attributions exactly equals f(x) - f(x_bg) (Shapley Efficiency Axiom).
        """
        # Encode patient vitals into 22-dimensional feature space
        x_patient = encode_clinical_features(raw_patient_dict).squeeze(0)  # shape (22,)

        # Compute point predictions
        with torch.no_grad():
            x_t = torch.from_numpy(x_patient).unsqueeze(0)
            bg_t = torch.from_numpy(self.bg_vector).unsqueeze(0)
            patient_prob = float(torch.sigmoid(self.model(x_t)).item())
            base_prob = float(torch.sigmoid(self.model(bg_t)).item())

        delta_total = patient_prob - base_prob

        # Grouped Permutation Coalitional SHAP with antithetic permutation pairs
        M = len(self.clinical_concepts)
        rng = np.random.default_rng(seed)
        all_z: List[np.ndarray] = []
        perm_orders: List[np.ndarray] = []

        half_k = max(n_permutations // 2, 1)
        for _ in range(half_k):
            p = rng.permutation(M)
            p_rev = p[::-1]
            for perm in [p, p_rev]:
                perm_orders.append(perm)
                z = self.bg_vector.copy()
                for c_idx in perm:
                    c_name = self.clinical_concepts[c_idx]
                    cols = CONCEPT_INDICES[c_name]
                    z[cols] = x_patient[cols]
                    all_z.append(z.copy())

        # Vectorized batch evaluation across all coalitional evaluations
        Z_tensor = torch.from_numpy(np.array(all_z, dtype=np.float32))
        with torch.no_grad():
            preds = torch.sigmoid(self.model(Z_tensor)).squeeze().numpy()

        preds = preds.reshape(len(perm_orders), M)
        shap_values = np.zeros(M, dtype=np.float64)

        for i, perm in enumerate(perm_orders):
            v_prev = base_prob
            for j, c_idx in enumerate(perm):
                v_curr = float(preds[i, j])
                shap_values[c_idx] += (v_curr - v_prev)
                v_prev = v_curr

        shap_values /= len(perm_orders)

        # Enforce exact floating-point efficiency axiom closure
        # (Residual is typically < 1e-15, but closing guarantees mathematical precision)
        residual = delta_total - float(np.sum(shap_values))
        shap_values += residual / float(M)

        attributions = []
        for idx, concept in enumerate(self.clinical_concepts):
            impact_pct = float(shap_values[idx] * 100.0)
            val = raw_patient_dict.get(concept, 0)
            attributions.append({
                "feature": concept,
                "label": FRIENDLY_FEATURE_NAMES.get(concept, concept),
                "value": val,
                "impact": round(impact_pct, 2),
                "raw_shap": float(shap_values[idx]),
                "is_risk_increasing": impact_pct > 0,
            })

        # Sort by absolute impact magnitude
        attributions.sort(key=lambda item: abs(item["impact"]), reverse=True)

        risk_category = (
            "High Risk" if patient_prob >= 0.70
            else ("Moderate Risk" if patient_prob >= 0.40 else "Low Risk")
        )

        return {
            "predicted_risk_percent": round(patient_prob * 100.0, 1),
            "baseline_risk_percent": round(base_prob * 100.0, 1),
            "efficiency_delta_percent": round(delta_total * 100.0, 2),
            "sum_attributions_percent": round(float(np.sum(shap_values) * 100.0), 2),
            "risk_category": risk_category,
            "attributions": attributions,
        }


if __name__ == "__main__":
    explainer = ClinicalSHAPExplainer()
    sample_patient = {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 280,
        "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1,
        "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
    }
    explanation = explainer.explain_patient(sample_patient)
    print(f"Predicted Cardiac Risk: {explanation['predicted_risk_percent']}% ({explanation['risk_category']})")
    print(f"Efficiency Delta: {explanation['efficiency_delta_percent']}% | Sum Attributions: {explanation['sum_attributions_percent']}%")
    print("Top 3 Risk Drivers:")
    for a in explanation["attributions"][:3]:
        print(f"  {a['label']} ({a['value']}): {a['impact']:+.2f}%")

