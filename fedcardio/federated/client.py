"""
FedCardio Sovereign Hospital Client
Implements Flower client protocol with strict local directory isolation,
FedProx proximal loss, and Differential Privacy (DP-SGD) execution.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import flwr as fl

from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.data.loader import get_hospital_loaders
from fedcardio.federated.differential_privacy import DifferentialPrivacyEngine
from fedcardio.evaluation.metrics import compute_clinical_metrics
from fedcardio.database.audit_db import log_client_metric, log_packet_inspection

from fedcardio.config import TOTAL_FEATURES

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, local_epochs: int = 3, lr: float = 0.01,
                 strategy: str = "FedProx", mu: float = 0.01, use_dp: bool = True,
                 epsilon: float = 3.0, delta: float = 1e-4, max_grad_norm: float = 1.0,
                 batch_size: int = 16, device: str = "cpu"):
        self.hospital_id = hospital_id
        self.local_epochs = local_epochs
        self.lr = lr
        self.strategy = strategy
        self.mu = mu
        self.use_dp = use_dp
        self.device = torch.device(device)
        
        # Load local isolated dataset
        self.train_loader, self.val_loader, self.n_samples = get_hospital_loaders(
            hospital_id=hospital_id, batch_size=batch_size
        )
        
        # Initialize local clinical model
        self.model = ClinicalNet(in_features=TOTAL_FEATURES).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # DP engine
        self.dp_engine = DifferentialPrivacyEngine(
            max_grad_norm=max_grad_norm, epsilon=epsilon, delta=delta
        ) if use_dp else None
        
    def get_parameters(self, config: Dict[str, Any] = None) -> List[np.ndarray]:
        return self.model.get_weights()
        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.set_weights(parameters)
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        start_time = time.time()
        round_id = config.get("current_round", 1)
        self.set_parameters(parameters)
        
        # Keep frozen copy of global parameters for FedProx proximal regularization
        global_params = [p.detach().clone() for p in self.model.parameters()]
        
        self.model.train()
        # Production standard SGD with momentum and weight decay
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        
        epoch_losses = []
        last_grad_norm = 0.0
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                # 1. Compute DP gradients strictly on clinical loss
                if self.use_dp and self.dp_engine is not None:
                    # Evaluate loss for telemetry
                    with torch.no_grad():
                        pred_logits = self.model(X_batch)
                        batch_loss = self.criterion(pred_logits, y_batch).item()
                        batch_losses.append(batch_loss)
                    
                    last_grad_norm = self.dp_engine.compute_and_assign_dp_gradients(
                        model=self.model,
                        criterion=self.criterion,
                        x_batch=X_batch,
                        y_batch=y_batch
                    )
                else:
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    batch_losses.append(loss.item())
                    loss.backward()
                    last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
                    
                # 2. Decouple FedProx proximal regularization from DP:
                # Add proximal penalty mu * (p.data - p_global.data) directly to p.grad
                if self.strategy == "FedProx" and self.mu > 0:
                    for param, g_param in zip(self.model.parameters(), global_params):
                        if param.grad is not None:
                            param.grad.data.add_(self.mu * (param.data - g_param.data))
                            
                # 3. Parameter update step
                optimizer.step()
            epoch_losses.append(np.mean(batch_losses) if batch_losses else 0.0)
            
        train_loss = float(np.mean(epoch_losses))
        
        # Evaluate locally on validation split
        val_loss, val_metrics = self._evaluate_local()
        duration = round(time.time() - start_time, 2)
        
        # Log to audit DB
        log_client_metric(
            round_id=round_id,
            hospital_id=self.hospital_id,
            n_samples=self.n_samples,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auc=val_metrics["roc_auc"],
            grad_norm=float(last_grad_norm),
            duration_sec=duration
        )
        
        # Packet telemetry: prove strictly weights (tensors) are transmitted, 0 patient records
        updated_weights = self.model.get_weights()
        payload_bytes = sum(w.nbytes for w in updated_weights)
        log_packet_inspection(
            round_id=round_id,
            hospital_id=self.hospital_id,
            direction="CLIENT_TO_SERVER",
            tensor_count=len(updated_weights),
            payload_bytes=payload_bytes
        )
        
        return updated_weights, self.n_samples, {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_metrics["roc_auc"],
            "val_sensitivity": val_metrics["sensitivity"]
        }
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        val_loss, val_metrics = self._evaluate_local()
        return val_loss, len(self.val_loader.dataset), val_metrics
        
    def _evaluate_local(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        losses = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                losses.append(loss.item())
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
        metrics = compute_clinical_metrics(np.array(all_targets), np.array(all_probs))
        return float(np.mean(losses)), metrics
