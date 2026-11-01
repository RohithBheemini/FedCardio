"""
Automated Test Suite: Privacy Defense & Clinical Explainability
Verifies gradient inversion defense (DLG) with DP-SGD and SHAP feature attribution.
"""
import pytest
import torch
import numpy as np

from fedcardio.federated.differential_privacy import DifferentialPrivacyEngine
from fedcardio.evaluation.attack_simulation import simulate_gradient_inversion_attack
from fedcardio.explainability.explainer import ClinicalSHAPExplainer
from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.config import TOTAL_FEATURES

def test_dp_engine_clipping_and_noise():
    """Verify that DP-SGD computes per-sample clipped gradients and accounts privacy."""
    model = ClinicalNet(in_features=TOTAL_FEATURES)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(8, TOTAL_FEATURES) * 5.0
    y = torch.tensor([[0.], [1.], [0.], [1.], [1.], [0.], [0.], [1.]])
    
    dp = DifferentialPrivacyEngine(max_grad_norm=0.5, epsilon=2.0, delta=1e-4)
    mean_norm = dp.compute_and_assign_dp_gradients(model, criterion, x, y)
    assert mean_norm > 0.0
    
    # Verify gradients were assigned to all trainable weights
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing for parameter {name}"
        assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert not torch.isinf(p.grad).any(), f"Inf in gradient for {name}"
        
    # Verify GDP privacy accountant
    eps_1, delta_1 = dp.compute_spent_privacy(current_round=1)
    eps_2, delta_2 = dp.compute_spent_privacy(current_round=2)
    assert eps_1 > 0.0
    assert eps_2 > eps_1, "Privacy budget should strictly accumulate across rounds"
    assert delta_1 == 1e-4

def test_gradient_inversion_defense():
    """Verify that DP-SGD prevents patient vitals reconstruction."""
    res_no_dp = simulate_gradient_inversion_attack(use_dp=False, steps=20)
    res_dp = simulate_gradient_inversion_attack(use_dp=True, steps=20)
    
    assert "final_mse" in res_no_dp
    assert "final_mse" in res_dp
    # DP final MSE should be significantly higher than unprotected final MSE
    assert res_dp["final_mse"] > res_no_dp["final_mse"]

def test_clinical_shap_explainer():
    """Verify that SHAP explainer returns patient risk % and satisfies Shapley Efficiency Axiom."""
    explainer = ClinicalSHAPExplainer()
    sample_patient = {
        "age": 60, "sex": 1, "cp": 4, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 145, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 1, "thal": 3
    }
    explanation = explainer.explain_patient(sample_patient, n_permutations=60)
    assert 0.0 <= explanation["predicted_risk_percent"] <= 100.0
    assert explanation["risk_category"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert len(explanation["attributions"]) == 13
    assert "impact" in explanation["attributions"][0]
    
    # Verify Shapley Efficiency Axiom: sum(attributions) == f(x) - f(x_bg)
    diff = abs(explanation["efficiency_delta_percent"] - explanation["sum_attributions_percent"])
    assert diff <= 0.05, f"Shapley efficiency violated: delta={explanation['efficiency_delta_percent']} vs sum={explanation['sum_attributions_percent']}"
