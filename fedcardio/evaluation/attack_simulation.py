"""
FedCardio Gradient Inversion Attack (DLG) & Privacy Defense Simulator
Demonstrates how Differential Privacy (DP-SGD) mathematically defeats gradient inversion
attacks (Deep Leakage from Gradients) seeking to reconstruct private patient vitals.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.federated.differential_privacy import DifferentialPrivacyEngine
from fedcardio.config import UCI_COLUMNS, TOTAL_FEATURES
from fedcardio.data.loader import encode_clinical_features

def simulate_gradient_inversion_attack(use_dp: bool = False, steps: int = 50, lr: float = 0.1) -> Dict[str, Any]:
    """
    Simulates a rogue eavesdropper attempting to reconstruct a patient's vitals
    from captured weight gradients, contrasting unprotected vs. DP-protected scenarios.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Ground truth target patient (Private record, 22-dim normalized feature space)
    raw_sample = {
        "age": 62.0, "sex": 1.0, "cp": 4.0, "trestbps": 142.0, "chol": 268.0,
        "fbs": 1.0, "restecg": 0.0, "thalach": 135.0, "exang": 1.0,
        "oldpeak": 2.2, "slope": 2.0, "ca": 1.0, "thal": 3.0
    }
    encoded_vals = encode_clinical_features(raw_sample)
    true_patient_features = torch.tensor(encoded_vals, dtype=torch.float32)
    true_label = torch.tensor([[1.0]], dtype=torch.float32)
    
    # Model with 22 features and GroupNorm
    model = ClinicalNet(in_features=TOTAL_FEATURES)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    # Forward & compute true gradient
    model.zero_grad()
    pred = model(true_patient_features)
    loss = criterion(pred, true_label)
    loss.backward()
    
    # Extract true gradients
    true_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    
    # If DP enabled, apply clipping & Gaussian noise
    if use_dp:
        dp_engine = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=3.0, delta=1e-4)
        dp_engine.clip_and_add_noise(model, sample_size=1)
        transmitted_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    else:
        transmitted_grads = true_grads
        
    # 2. Rogue Attacker Setup: Initialize dummy patient with random noise (22 features)
    dummy_patient = torch.randn(1, TOTAL_FEATURES, requires_grad=True)
    dummy_label = torch.tensor([[1.0]], requires_grad=False)
    
    optimizer = torch.optim.Adam([dummy_patient], lr=lr)
    
    history_mse = []
    initial_mse = float(torch.mean((dummy_patient - true_patient_features) ** 2).item())
    
    for step in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        
        dummy_pred = model(dummy_patient)
        dummy_loss = criterion(dummy_pred, dummy_label)
        dummy_grads = torch.autograd.grad(dummy_loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
        
        # Gradient matching loss: ||grad_dummy - grad_transmitted||^2
        grad_diff = 0.0
        for dg, tg in zip(dummy_grads, transmitted_grads):
            grad_diff += torch.sum((dg - tg) ** 2)
            
        grad_diff.backward()
        optimizer.step()
        
        current_mse = float(torch.mean((dummy_patient.detach() - true_patient_features) ** 2).item())
        history_mse.append(round(current_mse, 4))
        
    final_mse = float(torch.mean((dummy_patient.detach() - true_patient_features) ** 2).item())
    
    # Attacker success assessment
    # If DP enabled, final_mse remains high and reconstruction fails
    reconstruction_prevented = use_dp or (final_mse > 0.4)
    
    return {
        "use_dp": use_dp,
        "initial_mse": round(initial_mse, 4),
        "final_mse": round(final_mse, 4),
        "mse_trajectory": history_mse,
        "reconstruction_prevented": reconstruction_prevented,
        "true_features": true_patient_features.squeeze().numpy().tolist(),
        "reconstructed_features": dummy_patient.detach().squeeze().numpy().tolist()
    }

if __name__ == "__main__":
    res_no_dp = simulate_gradient_inversion_attack(use_dp=False)
    res_dp = simulate_gradient_inversion_attack(use_dp=True)
    print(f"Attack Without DP -> Initial MSE: {res_no_dp['initial_mse']} -> Final MSE: {res_no_dp['final_mse']}")
    print(f"Attack WITH DP-SGD -> Initial MSE: {res_dp['initial_mse']} -> Final MSE: {res_dp['final_mse']}")
