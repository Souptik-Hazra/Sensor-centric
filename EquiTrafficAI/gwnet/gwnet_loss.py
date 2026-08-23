"""
EquiTraffic-GPT MLOps Module 2: Loss & Metrics Layer (gwnet_loss.py)

Modern PyTorch 2.x Loss & Evaluation Metrics:
- SmartRerouteLoss: Dynamic speed threshold calibration + native torch.diff() for temporal trend derivatives
- calculate_r2_score: Coefficient of determination (R²) evaluation benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calculate_r2_score(y_true, y_pred):
    """Calculates R2 Score (Coefficient of Determination)."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-8)))


class SmartRerouteLoss(nn.Module):
    """
    Purpose-Driven Custom Loss Function for 15-Minute Highway Smart Rerouting.
    
    Combines:
    1. Base L1 Loss (Global MAE across all speed ranges)
    2. Bottleneck Penalty: Heavily penalizes prediction errors during severe bottleneck slowdowns
    3. Speed Derivative Loss (torch.diff): Forces GWNet to accurately predict rapid deceleration trend rates (dv/dt)
    """
    def __init__(self, alpha=3.0, beta=1.5, speed_threshold_norm=-1.5):
        super(SmartRerouteLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.speed_threshold_norm = speed_threshold_norm

    def forward(self, y_pred, y_true):
        base_loss = F.l1_loss(y_pred, y_true)
        
        # Dynamic bottleneck penalty mask for severe slowdowns below threshold
        bottleneck_mask = F.relu(self.speed_threshold_norm - y_true)
        bottleneck_penalty = torch.mean(bottleneck_mask * torch.abs(y_pred - y_true))
        
        # PyTorch 1.8+ native torch.diff for temporal speed trend derivatives
        if y_pred.size(1) > 1:
            pred_diff = torch.diff(y_pred, dim=1)
            true_diff = torch.diff(y_true, dim=1)
            diff_loss = F.l1_loss(pred_diff, true_diff)
        else:
            diff_loss = torch.tensor(0.0, device=y_pred.device)

        return base_loss + self.alpha * bottleneck_penalty + self.beta * diff_loss
