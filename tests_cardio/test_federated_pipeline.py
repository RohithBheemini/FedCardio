"""
Automated Test Suite: Federated Learning Pipeline & Aggregation
Verifies model serialization, client local training, FedProx aggregation, and round convergence.
"""
import pytest
import torch
import numpy as np

from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.federated.client import HospitalClient
from fedcardio.federated.strategies import aggregate_weights, compute_client_drift
from fedcardio.federated.server import FederatedConsortiumServer
from fedcardio.config import FLConfig, TOTAL_FEATURES

def test_clinical_net_forward_and_payload():
    """Verify PyTorch model forward pass, dimensions, and payload calculation."""
    model = ClinicalNet(in_features=TOTAL_FEATURES)
    dummy_input = torch.randn(8, TOTAL_FEATURES)
    out = model(dummy_input)
    assert out.shape == (8, 1)
    
    probs = model.predict_proba(dummy_input)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    payload_kb = model.calculate_payload_size_kb()
    assert 10.0 < payload_kb < 100.0, f"Payload size unexpected: {payload_kb} KB"

def test_hospital_client_fit_and_drift():
    """Verify HospitalClient local training with FedProx and drift calculation."""
    client = HospitalClient(hospital_id="cleveland", local_epochs=1, strategy="FedProx", mu=0.01)
    initial_weights = client.get_parameters()
    
    new_weights, n_samples, metrics = client.fit(initial_weights, config={"current_round": 1})
    assert n_samples > 100
    assert "train_loss" in metrics
    assert "val_auc" in metrics
    
    drift = compute_client_drift(initial_weights, new_weights)
    assert drift >= 0.0

def test_federated_consortium_2_rounds():
    """Verify end-to-end 2-round federated training run."""
    config = FLConfig(num_rounds=2, local_epochs=1, strategy="FedProx", use_dp=True, epsilon=3.0)
    server = FederatedConsortiumServer(config)
    history = server.run_consortium_rounds(num_rounds=2)
    
    assert len(history) == 2
    assert "roc_auc" in history[0]
    assert "epsilon_spent" in history[0]
    assert history[1]["roc_auc"] > 0.50
