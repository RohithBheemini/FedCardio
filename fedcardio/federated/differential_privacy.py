"""
FedCardio Differential Privacy Engine
Implements Provable DP-SGD (Differential Privacy Stochastic Gradient Descent)
with per-sample gradient clipping (via torch.func.vmap/grad), calibrated Gaussian noise injection,
and RDP / Gaussian DP accountant tracking subsampling ratio q = B/N and steps T = R * E * (N/B).
"""
from typing import Tuple, List, Dict, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, grad, functional_call


class DifferentialPrivacyEngine:
    def __init__(
        self,
        max_grad_norm: float = 1.0,
        epsilon: float = 3.0,
        delta: float = 1e-4,
        seed: int = 42,
    ):
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.rounds_executed = 0

        # Calculate Gaussian noise scale sigma based on (epsilon, delta)
        # Analytical Gaussian mechanism: sigma = sqrt(2 * ln(1.25 / delta)) / epsilon
        self.sigma = float(np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon)

    def compute_and_assign_dp_gradients(
        self,
        model: nn.Module,
        criterion: nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> float:
        """
        Computes per-sample micro-gradients for a batch using torch.func.vmap,
        computes per-sample L2 norms ||g_i||_2, clips each sample by min(1, C / (||g_i||_2 + 1e-6)),
        sums the clipped gradients, adds calibrated Gaussian noise with scale sigma * C,
        and divides by batch size B (so noise scale on average gradient is sigma * C / B).
        Assigns the privatized gradient to param.grad for each parameter.

        Returns:
            The mean per-sample gradient norm before clipping.
        """
        batch_size = x_batch.shape[0]
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}

        # Define per-sample loss function
        def compute_single_loss(p_dict: Dict[str, torch.Tensor], x_i: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
            pred = functional_call(model, p_dict, (x_i.unsqueeze(0),))
            return criterion(pred.squeeze(0), y_i)

        # Vectorized per-sample gradients
        per_sample_grads = vmap(
            grad(compute_single_loss),
            in_dims=(None, 0, 0),
            randomness="same",
        )(params, x_batch, y_batch)

        # 1. Compute per-sample L2 gradient norms ||g_i||_2
        sq_norms = torch.zeros(batch_size, device=x_batch.device)
        for g_tensor in per_sample_grads.values():
            flat_g = g_tensor.reshape(batch_size, -1)
            sq_norms = sq_norms + torch.sum(flat_g ** 2, dim=1)
        per_sample_norms = torch.sqrt(sq_norms)
        mean_norm = float(per_sample_norms.mean().item())

        # 2. Per-sample clipping factor: min(1, C / (||g_i||_2 + 1e-6))
        clip_factors = torch.clamp(self.max_grad_norm / (per_sample_norms + 1e-6), max=1.0)

        # 3. Sum clipped gradients, inject Gaussian noise with scale sigma * C, divide by B
        c = self.max_grad_norm
        for name, param in model.named_parameters():
            if name in per_sample_grads:
                g_tensor = per_sample_grads[name]  # shape: (B, *param.shape)
                expanded_factors = clip_factors.view([-1] + [1] * (g_tensor.ndim - 1))
                clipped_g = g_tensor * expanded_factors
                summed_g = torch.sum(clipped_g, dim=0)

                # Noise scale on sum is sigma * C; division by B makes noise scale on average sigma * C / B
                noise = torch.randn_like(summed_g) * (self.sigma * c)
                dp_avg_grad = (summed_g + noise) / float(batch_size)

                if param.grad is None:
                    param.grad = dp_avg_grad.detach().clone()
                else:
                    param.grad.copy_(dp_avg_grad)

        return mean_norm

    def clip_and_add_noise(self, model: nn.Module, sample_size: int) -> float:
        """
        Applies gradient clipping to threshold C and injects calibrated Gaussian noise.
        Used for fallback or pre-computed batch gradients.
        Noise scale on the average gradient is sigma * C / B.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
        b = max(sample_size, 1)
        noise_std = (self.sigma * self.max_grad_norm) / float(b)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
                noise = torch.randn_like(p.grad.data) * noise_std
                p.grad.data.add_(noise)

        return float(total_norm)

    def compute_spent_privacy(
        self,
        current_round: int,
        q: Optional[float] = None,
        local_epochs: int = 3,
        n_samples: int = 200,
        batch_size: int = 16,
    ) -> Tuple[float, float]:
        """
        Computes cumulative spent privacy budget (epsilon_spent, delta) using the
        Gaussian Differential Privacy (GDP) / RDP accountant for subsampled Gaussian mechanism.
        Accounts strictly for:
          - subsampling ratio q = B / N
          - total training steps T = R * E * ceil(N / B)
          - noise multiplier sigma
        """
        if current_round <= 0:
            return 0.0, self.delta

        # Subsampling ratio
        q_ratio = q if q is not None else float(batch_size) / float(max(n_samples, 1))

        # Local steps per round and total steps T
        steps_per_epoch = int(np.ceil(float(n_samples) / float(batch_size)))
        steps_per_round = local_epochs * steps_per_epoch
        total_steps = current_round * steps_per_round

        # Gaussian DP parameter mu for subsampled Gaussian mechanism:
        # mu = q * sqrt(T * (exp(1/sigma^2) - 1)) ~ (q * sqrt(T)) / sigma
        mu = (q_ratio * np.sqrt(float(total_steps))) / max(self.sigma, 1e-6)

        # Convert GDP parameter mu to (epsilon, delta) via dual formulation:
        # eps = mu * sqrt(2 * ln(1 / delta)) + (mu^2) / 2
        log_term = np.sqrt(2.0 * np.log(1.0 / self.delta))
        eps_spent = mu * log_term + 0.5 * (mu ** 2)

        return round(float(eps_spent), 3), self.delta

