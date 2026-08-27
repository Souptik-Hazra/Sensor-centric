"""
EquiTraffic-GPT MLOps Module 4: High-Performance Model Trainer Pipeline (gwnet_trainer.py)

Modern PyTorch 2.x Accelerated Model Trainer & Checkpoint Engine:
- SOTA Architecture Support (Spatial-Temporal FlashAttention + LayerNorm Residual Stabilization)
- Dynamic Hyperparameter Resolution from model_config.yaml
- Dual Checkpoint & Model Versioning in model_registry.json
"""

import os
import sys
import time
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from gwnet_model import GraphWaveNet
from gwnet_loss import SmartRerouteLoss, calculate_r2_score
from gwnet_dataset import build_adj_matrix_from_distances, load_pems_sequences, load_pems_adjacency
from gwnet_registry import get_next_version, register_model_version

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


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


def train_full_gwnet(dataset_name="metr_la", num_epochs=None, batch_size=None, lr=None, stride=None, use_amp=False, resume=False, version=None, use_compile=False, use_attn=True):
    print("=================================================================")
    print(f"   EQUITRAFFIC-GPT SOTA MLOPS TRAINER ({dataset_name.upper()}) ")
    print("=================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Compute Hardware Accelerator: {device.type.upper()}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    cfg = load_model_config()
    gnn_cfg = cfg.get('graph_wavenet_gnn', {})
    arch_cfg = gnn_cfg.get('architecture', {})

    clean_ds = dataset_name.lower().replace("_", "").replace("-", "")
    num_epochs = num_epochs if num_epochs is not None else gnn_cfg.get('training', {}).get('default_epochs', 50)
    batch_size = batch_size if batch_size is not None else gnn_cfg.get('training', {}).get('default_batch_size', 64)
    lr = lr if lr is not None else gnn_cfg.get('training', {}).get('default_learning_rate', 0.001)
    stride = stride if stride is not None else gnn_cfg.get('training', {}).get('default_stride', 2)

    his_path = os.path.join(data_dir, 'metr_la_his.npz') if 'la' in clean_ds else os.path.join(data_dir, 'sd400_his.npz')

    X_tr, Y_tr, X_val, Y_val, speed_mean, speed_std, num_nodes = load_pems_sequences(his_path, stride=stride)

    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adj_matrix = load_pems_adjacency(data_dir, dataset_name)
    adj_tensor = torch.FloatTensor(adj_matrix).to(device)
    supports = [adj_tensor]

    threshold_mph = gnn_cfg.get('training', {}).get('bottleneck_threshold_mph', 25.0)
    threshold_norm = (threshold_mph - speed_mean) / speed_std if speed_std > 0 else -1.5

    out_dim = 12 if "sd" in clean_ds else 1
    skip_ch = 64 if "sd" in clean_ds else arch_cfg.get('skip_channels', 256)
    end_ch = 128 if "sd" in clean_ds else arch_cfg.get('end_channels', 512)

    raw_model = GraphWaveNet(
        num_nodes=num_nodes,
        in_dim=gnn_cfg.get('input_dimensions', 3),
        out_dim=out_dim,
        horizon=12,
        supports=supports,
        adp_adj=True,
        residual_channels=arch_cfg.get('residual_channels', 32),
        dilation_channels=arch_cfg.get('dilation_channels', 32),
        skip_channels=skip_ch,
        end_channels=end_ch,
        dropout=arch_cfg.get('dropout', 0.3),
        use_attn=use_attn
    ).to(device)

    model = torch.compile(raw_model) if use_compile and hasattr(torch, 'compile') else raw_model

    loss_cfg = gnn_cfg.get('loss', {})
    criterion = SmartRerouteLoss(alpha=loss_cfg.get('alpha', 3.0), beta=loss_cfg.get('beta', 1.5), speed_threshold_norm=threshold_norm)
    optimizer = optim.Adam(raw_model.parameters(), lr=lr, weight_decay=gnn_cfg.get('training', {}).get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=gnn_cfg.get('training', {}).get('eta_min', 1e-5))

    version_str = version if version else get_next_version(dataset_name)
    print(f"[+] MLOps Registry: Target Version Allocated -> '{version_str}' ({num_nodes} Nodes)")

    best_val_mae = float('inf')
    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        raw_model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            x_b = x_b.permute(0, 3, 2, 1)

            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=gnn_cfg.get('training', {}).get('max_grad_norm', 5.0))
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_mae = train_loss / len(train_loader)

        raw_model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                x_b = x_b.permute(0, 3, 2, 1)
                out = model(x_b)
                loss = criterion(out, y_b)
                val_loss += loss.item()
                val_preds.append(out.cpu().numpy())
                val_targets.append(y_b.cpu().numpy())

        val_preds_arr = np.concatenate(val_preds, axis=0)
        val_targets_arr = np.concatenate(val_targets, axis=0)

        # Un-normalize Z-scores to calculate true physical speeds in MPH
        real_preds_mph = val_preds_arr * speed_std + speed_mean
        real_targets_mph = val_targets_arr * speed_std + speed_mean

        true_mae_mph = float(np.mean(np.abs(real_preds_mph - real_targets_mph)))
        r2 = calculate_r2_score(torch.tensor(real_preds_mph), torch.tensor(real_targets_mph))
        epoch_sec = time.time() - t0

        print(f"Epoch {epoch:2d} | Train: {avg_train_mae:.4f} | Val Loss: {avg_val_mae:.4f} | Val MAE: {true_mae_mph:.2f} mph | R²: {r2:.4f} | {epoch_sec:.2f} s", end="")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            versioned_dir = os.path.join(base_dir, "checkpoints", version_str, dataset_name)
            os.makedirs(versioned_dir, exist_ok=True)
            
            versioned_pt_path = os.path.join(versioned_dir, f"gwnet_{dataset_name}_{version_str}.pt")
            versioned_tar_path = os.path.join(versioned_dir, f"gwnet_{dataset_name}_{version_str}.tar")

            torch.save(raw_model.state_dict(), versioned_pt_path)
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                'speed_mean': speed_mean,
                'speed_std': speed_std,
                'num_nodes': num_nodes,
                'use_attn': use_attn
            }, versioned_tar_path)

            metrics = {"val_mae": round(float(best_val_mae), 4), "val_mae_mph": round(float(val_mae_mph), 2), "val_r2": round(float(r2), 4)}
            hparams = {"in_dim": 3, "horizon": 12, "batch_size": batch_size, "lr": lr, "stride": stride, "alpha": loss_cfg.get('alpha', 3.0), "beta": loss_cfg.get('beta', 1.5), "use_attn": use_attn}
            register_model_version(dataset_name, version_str, versioned_pt_path, versioned_tar_path, metrics, hparams)
            status = f"[SAVED {version_str}]"
        else:
            status = ""

        log_msg = f"Epoch {epoch:<3} | Train: {avg_train_mae:<6.4f} | Val Norm: {avg_val_mae:<6.4f} | Val MPH: {val_mae_mph:<6.2f}mph | R²: {r2:<6.4f} | {epoch_sec:<6.2f}s | {status}\n"
        sys.stdout.write(log_msg)
        sys.stdout.flush()

    total_sec = time.time() - start_total_time
    print(f"\n[+] Total SOTA Accelerated Training Time ({num_epochs} Epochs): {total_sec / 60.0:.2f} minutes!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="metr_la")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_attn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_full_gwnet(args.dataset, args.epochs, stride=args.stride, use_amp=args.amp, resume=args.resume, version=args.version, use_compile=args.compile, use_attn=not args.no_attn)
