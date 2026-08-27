"""
EquiTraffic-GPT MLOps Module 2: Physics-Informed & Equity-Aware Loss Function (gwnet_loss.py)

Contains customized PyTorch 2.x loss modules & evaluation metrics:
- SmartRerouteLoss: Combines MAE with physics-informed asymmetric bottleneck speed penalty
- Masked MAE, MAPE, RMSE, and R2 determination coefficient evaluation metrics
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model_config() -> dict:
    """Load model_config.yaml from backend directory with robust fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, '..', 'backend', 'model_config.yaml'),
        os.path.join(base_dir, 'model_config.yaml')
    ]
    for cfg_path in candidates:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
    return {}


class SmartRerouteLoss(nn.Module):
    """
    Physics-Informed & Bottleneck Penalty Loss Function.
    Applies asymmetric exponential penalty (alpha * exp(threshold - y_hat)) when predicted speed drops below bottleneck threshold.
    """
    def __init__(self, alpha=None, beta=None, speed_threshold_norm=None):
        super(SmartRerouteLoss, self).__init__()
        cfg_loss = load_model_config().get('graph_wavenet_gnn', {}).get('loss', {})
        self.alpha = alpha if alpha is not None else cfg_loss.get('alpha', 3.0)
        self.beta = beta if beta is not None else cfg_loss.get('beta', 1.5)
        self.speed_threshold_norm = speed_threshold_norm if speed_threshold_norm is not None else cfg_loss.get('default_speed_threshold_norm', -1.5)

    def forward(self, y_pred, y_true):
        # 1. Base Masked Mean Absolute Error (MAE)
        mask = (y_true != 0).float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        base_mae = torch.abs(y_pred - y_true)
        base_mae = base_mae * mask
        base_mae = torch.where(torch.isnan(base_mae), torch.zeros_like(base_mae), base_mae)
        
        # 2. Physics-Informed Bottleneck Underestimation Penalty
        bottleneck_mask = (y_true < self.speed_threshold_norm).float()
        underpredict_error = F.relu(y_true - y_pred) * bottleneck_mask
        penalty = self.alpha * torch.exp(underpredict_error) - self.alpha
        
        loss = torch.mean(base_mae) + self.beta * torch.mean(penalty * mask)
        return loss


def calculate_r2_score(y_pred, y_true):
    """Calculates R^2 Determination Coefficient metric."""
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 1.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2.item())
