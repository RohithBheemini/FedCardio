"""
FedCardio Federated Server & 8-Hospital Consortium Orchestrator
Coordinates multi-hospital federated training rounds, executes standard and Byzantine-resilient
aggregation (FedAvg, FedProx, FedMedian, TrimmedMean), tracks per-hospital validation trajectories
and directional cosine drift, evaluates against the 8-center holdout cohort, and tracks privacy budgets.
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from fedcardio.config import HOSPITALS, FLConfig, CHECKPOINT_DIR, TOTAL_FEATURES
from fedcardio.models.clinical_net import ClinicalNet
from fedcardio.federated.client import HospitalClient
from fedcardio.federated.strategies import (
    aggregate_weights,
    compute_client_drift,
    compute_cosine_drift,
    simulate_byzantine_attack,
)
from fedcardio.federated.differential_privacy import DifferentialPrivacyEngine
from fedcardio.data.loader import get_global_evaluation_loader
from fedcardio.evaluation.metrics import compute_clinical_metrics
from fedcardio.database.audit_db import (
    log_fl_round,
    log_packet_inspection,
    log_model_checkpoint,
    init_database,
    get_max_round_id,
)


class FederatedConsortiumServer:
    def __init__(self, config: Optional[FLConfig] = None):
        self.config = config or FLConfig()
        init_database()

        self.device = torch.device(self.config.device)
        self.global_model = ClinicalNet(in_features=TOTAL_FEATURES).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        # Dynamically initialize current_round from persistent SQLite audit database
        self.current_round = get_max_round_id()
        if self.current_round > 0:
            latest_ckpt = CHECKPOINT_DIR / f"global_model_round_{self.current_round}.pt"
            if not latest_ckpt.exists():
                latest_ckpt = CHECKPOINT_DIR / "best_model.pt"
            if latest_ckpt.exists():
                try:
                    self.global_model.load_state_dict(
                        torch.load(latest_ckpt, map_location=self.device)
                    )
                except Exception:
                    pass

        # Ensure isolated hospital datasets exist before initializing clients
        from fedcardio.data.downloader import fetch_or_generate_hospital_data
        fetch_or_generate_hospital_data(force=False)

        # Initialize clients strictly bound to their isolated physical folders
        self.clients = {
            h_id: HospitalClient(
                hospital_id=h_id,
                local_epochs=self.config.local_epochs,
                lr=self.config.learning_rate,
                strategy=self.config.strategy,
                mu=self.config.mu,
                use_dp=self.config.use_dp,
                epsilon=self.config.epsilon,
                delta=self.config.delta,
                max_grad_norm=self.config.max_grad_norm,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
            for h_id in HOSPITALS.keys()
        }

        self.dp_engine = DifferentialPrivacyEngine(
            max_grad_norm=self.config.max_grad_norm,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
        )

        # Load multi-center holdout cohort
        self.eval_loader, self.eval_df = get_global_evaluation_loader()

    def run_consortium_rounds(
        self,
        num_rounds: Optional[int] = None,
        callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        rounds_to_run = num_rounds or self.config.num_rounds
        start_round = self.current_round + 1
        end_round = self.current_round + rounds_to_run

        print("\n=======================================================")
        print(f"Starting FedCardio 8-Hospital Consortium Training ({rounds_to_run} rounds: #{start_round} -> #{end_round})")
        print(f"Strategy: {self.config.strategy} | Privacy: DP-SGD (epsilon={self.config.epsilon}) | Byzantine Nodes: {self.config.byzantine_nodes}")
        print(f"Participating Hospitals ({len(self.clients)}): {list(self.clients.keys())}")
        print("=======================================================\n")

        history = []
        best_auc = 0.0
        hospital_ids = list(self.clients.keys())

        for r in range(start_round, end_round + 1):
            round_start = time.time()
            self.current_round = r
            global_weights = self.global_model.get_weights()
            payload_kb = self.global_model.calculate_payload_size_kb()

            # Broadcast to clients & train locally inside isolated hospital silos
            client_weights = []
            sample_counts = []
            client_drifts = {}
            client_cosine_drifts = {}
            hospital_val_aucs = {}

            step_idx = r - start_round + 1
            print(f"--- Round {r} (Step {step_idx}/{rounds_to_run}) ---")
            for idx, (h_id, client) in enumerate(self.clients.items()):
                fit_res, n_samples, metrics = client.fit(
                    global_weights, config={"current_round": r}
                )

                # Optional Byzantine poisoning simulation on the last `byzantine_nodes` clients
                is_poisoned = idx >= (len(hospital_ids) - self.config.byzantine_nodes)
                if is_poisoned and self.config.byzantine_nodes > 0:
                    fit_res = simulate_byzantine_attack(
                        fit_res, attack_type="sign_flip", scale=4.5, seed=r * 10 + idx
                    )

                client_weights.append(fit_res)
                sample_counts.append(n_samples)

                drift_l2 = compute_client_drift(global_weights, fit_res)
                drift_cos = compute_cosine_drift(global_weights, fit_res)
                client_drifts[h_id] = round(drift_l2, 4)
                client_cosine_drifts[h_id] = round(drift_cos, 6)
                hospital_val_aucs[h_id] = round(metrics["val_auc"], 4)

                tag = " [POISONED]" if is_poisoned and self.config.byzantine_nodes > 0 else ""
                print(
                    f"  [{HOSPITALS[h_id]['name']}{tag}] {n_samples} pts | "
                    f"Loss: {metrics['train_loss']:.4f} | Val AUC: {metrics['val_auc']:.4f} | "
                    f"L2 Drift: {drift_l2:.3f} | CosDrift: {drift_cos:.4f}"
                )

            # Aggregate weights using selected strategy (FedAvg / FedProx / FedMedian / TrimmedMean)
            aggregated_weights = aggregate_weights(
                client_weights,
                sample_counts,
                strategy=self.config.strategy,
                trim_ratio=self.config.trim_ratio,
            )
            self.global_model.set_weights(aggregated_weights)

            # Server-side broadcast packet logging
            for h_id in self.clients.keys():
                log_packet_inspection(
                    round_id=r,
                    hospital_id=h_id,
                    direction="SERVER_TO_CLIENT",
                    tensor_count=len(aggregated_weights),
                    payload_bytes=int(payload_kb * 1024),
                )

            # Evaluate Global Consensus Model on Held-out 8-Center Cohort
            eval_loss, eval_metrics = self.evaluate_global_model()

            # Compute cumulative spent privacy budget
            eps_spent, delta_spent = self.dp_engine.compute_spent_privacy(r)

            # Log round to persistent SQLite audit store
            log_fl_round(
                round_id=r,
                strategy=self.config.strategy,
                num_clients=len(self.clients),
                global_loss=eval_loss,
                global_roc_auc=eval_metrics["roc_auc"],
                global_pr_auc=eval_metrics["pr_auc"],
                epsilon_spent=eps_spent,
                payload_kb=payload_kb,
            )

            # Save Checkpoint
            checkpoint_path = CHECKPOINT_DIR / f"global_model_round_{r}.pt"
            torch.save(self.global_model.state_dict(), checkpoint_path)

            if eval_metrics["roc_auc"] > best_auc:
                best_auc = eval_metrics["roc_auc"]
                torch.save(self.global_model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

            round_duration = round(time.time() - round_start, 2)
            round_summary = {
                "round": r,
                "strategy": self.config.strategy,
                "byzantine_nodes": self.config.byzantine_nodes,
                "loss": eval_loss,
                "roc_auc": eval_metrics["roc_auc"],
                "pr_auc": eval_metrics["pr_auc"],
                "sensitivity": eval_metrics["sensitivity"],
                "specificity": eval_metrics["specificity"],
                "brier_score": eval_metrics["brier_score"],
                "epsilon_spent": eps_spent,
                "payload_kb": payload_kb,
                "client_drifts": client_drifts,
                "client_cosine_drifts": client_cosine_drifts,
                "hospital_val_aucs": hospital_val_aucs,
                "duration_sec": round_duration,
            }
            log_model_checkpoint(
                round_id=r,
                file_path=str(checkpoint_path),
                metrics_json=json.dumps(round_summary),
            )
            history.append(round_summary)
            print(
                f"Round {r} Complete: Global ROC-AUC={eval_metrics['roc_auc']:.4f} | "
                f"PR-AUC={eval_metrics['pr_auc']:.4f} | DP eps={eps_spent:.2f} ({round_duration}s)\n"
            )

            if callback is not None:
                callback(step_idx, rounds_to_run, round_summary)

        return history

    def evaluate_global_model(self) -> Tuple[float, Dict[str, float]]:
        """Evaluates global model on multi-center evaluation cohort."""
        self.global_model.eval()
        losses = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.eval_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.global_model(X_batch)
                loss = self.criterion(logits, y_batch)
                losses.append(loss.item())
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        metrics = compute_clinical_metrics(np.array(all_targets), np.array(all_probs))
        return float(np.mean(losses)), metrics


def run_federated_pipeline(
    num_rounds: int = 5,
    strategy: str = "FedProx",
    use_dp: bool = True,
    epsilon: float = 3.0,
    byzantine_nodes: int = 0,
) -> List[Dict[str, Any]]:
    config = FLConfig(
        num_rounds=num_rounds,
        strategy=strategy,
        use_dp=use_dp,
        epsilon=epsilon,
        byzantine_nodes=byzantine_nodes,
    )
    server = FederatedConsortiumServer(config)
    return server.run_consortium_rounds()


if __name__ == "__main__":
    run_federated_pipeline(num_rounds=3)
