"""
EquiTraffic-GPT MLOps Module 4: High-Performance Model Trainer Pipeline (gwnet_trainer.py)

Modern PyTorch 2.x Accelerated Model Trainer & Checkpoint Engine:
- SOTA Architecture Support (Spatial-Temporal FlashAttention + LayerNorm Residual Stabilization)
- Optional PyTorch 2.0 torch.compile(model) Kernel Fusion Support (--compile)
- Anti-Data-Leakage Strict Normalization & Dual Physical Speed MAE Reporting (in mph)
- Dual Checkpoint & Model Versioning in model_registry.json
"""

import os
import sys
import time
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


def train_full_gwnet(dataset_name="metr_la", num_epochs=50, batch_size=64, lr=0.001, stride=2, use_amp=False, resume=False, version=None, use_compile=False, use_attn=True):
    print("=================================================================")
    print(f"   EQUITRAFFIC-GPT SOTA MLOPS TRAINER ({dataset_name.upper()}) ")
    print("=================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda_amp = (device.type == 'cuda') and use_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    version_str = version if version else get_next_version(dataset_name)
    print(f"[+] Execution Device: {device} | AMP FP16: {use_cuda_amp} | FlashAttention: {use_attn} | Target Version: {version_str}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints", version_str, dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, '..', 'data')
    his_path = os.path.join(data_dir, f'{dataset_name}_his.npz')

    start_prep_time = time.time()
    X_train, Y_train, X_val, Y_val, speed_mean, speed_std, num_nodes = load_pems_sequences(his_path, stride=stride)
    prep_elapsed = time.time() - start_prep_time
    print(f"[+] Vectorized Dataset Prep Completed in: {prep_elapsed:.2f} seconds!")

    # Calculate normalized threshold for 25.0 mph slowdown bottleneck penalty
    threshold_norm = (25.0 - speed_mean) / speed_std if speed_std > 1e-5 else -1.5

    # Load pre-computed binary spatial adjacency pickle in 0.001s
    adj_matrix = load_pems_adjacency(data_dir, dataset_name)
    adj_tensor = torch.FloatTensor(adj_matrix).to(device)
    supports = [adj_tensor]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    num_workers = 2 if os.name == 'posix' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = GraphWaveNet(
        num_nodes=num_nodes, in_dim=3, out_dim=1, horizon=12,
        supports=supports, adp_adj=True, dropout=0.3,
        residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
        use_attn=use_attn
    ).to(device)

    # PyTorch 2.0+ C++ Kernel Fusion compilation
    if use_compile and hasattr(torch, "compile"):
        try:
            print("[+] PyTorch 2.0: Compiling SOTA GWNet model with TorchDynamo (mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[!] PyTorch 2.0 Compile Notice: Falling back to standard execution: {e}")

    criterion = SmartRerouteLoss(alpha=3.0, beta=1.5, speed_threshold_norm=threshold_norm)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_mae = float('inf')
    start_epoch = 1

    versioned_pt_path = os.path.join(ckpt_dir, f'gwnet_{dataset_name}_{version_str}.pt')
    versioned_tar_path = os.path.join(ckpt_dir, f'gwnet_{dataset_name}_{version_str}.tar')
    
    base_pt_path = os.path.join(base_dir, f'gwnet_{dataset_name}.pt')
    base_pth_path = os.path.join(base_dir, f'gwnet_{dataset_name}_best.pth')

    if resume and os.path.exists(versioned_tar_path):
        print(f"[+] Resuming Training from Checkpoint Bundle: {versioned_tar_path}")
        ckpt = torch.load(versioned_tar_path, map_location=device)
        state = ckpt['model_state_dict']
        # Handle state dict prefix if compiled
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(state)
        else:
            model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_mae = ckpt.get('best_val_mae', float('inf'))
        print(f"[+] Resumed cleanly at Epoch {start_epoch} (Best Val MAE: {best_val_mae:.4f})")

    print(f"\n[+] Training SOTA GWNet [{version_str}] from Epoch {start_epoch} to {num_epochs} (Stride={stride}, Batch={batch_size})...\n")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val MAE (Norm)':<14} | {'Val MAE (mph)':<14} | {'Val R²':<8} | {'Time (s)':<8} | Status")
    print("-" * 90)

    start_total_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_mae_sum = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_cuda_amp):
                out = model(x_b)
                loss = criterion(out, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_mae_sum += loss.item()

        avg_train_mae = train_mae_sum / len(train_loader)
        scheduler.step()

        model.eval()
        val_mae_sum = 0.0
        val_preds, val_trues = [], []
        
        with torch.inference_mode():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                with torch.amp.autocast('cuda', enabled=use_cuda_amp):
                    out = model(x_b)
                    loss_val = criterion(out, y_b)
                val_mae_sum += loss_val.item()
                val_preds.append(out.cpu().numpy())
                val_trues.append(y_b.cpu().numpy())

        avg_val_mae = val_mae_sum / len(val_loader)
        val_mae_mph = avg_val_mae * speed_std
        r2 = calculate_r2_score(np.concatenate(val_trues, axis=0), np.concatenate(val_preds, axis=0))
        epoch_sec = time.time() - epoch_start

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            
            torch.save(raw_model.state_dict(), versioned_pt_path)
            torch.save(raw_model.state_dict(), base_pt_path)
            torch.save(raw_model.state_dict(), base_pth_path)
            
            torch.save({
                'version': version_str,
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'best_val_mae_mph': val_mae_mph,
                'best_val_r2': r2,
                'speed_mean': speed_mean,
                'speed_std': speed_std,
                'num_nodes': num_nodes,
                'use_attn': use_attn
            }, versioned_tar_path)

            metrics = {"val_mae": round(float(best_val_mae), 4), "val_mae_mph": round(float(val_mae_mph), 2), "val_r2": round(float(r2), 4)}
            hparams = {"in_dim": 3, "horizon": 12, "batch_size": batch_size, "lr": lr, "stride": stride, "alpha": 3.0, "beta": 1.5, "use_attn": use_attn}
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_attn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_full_gwnet(args.dataset, args.epochs, stride=args.stride, use_amp=args.amp, resume=args.resume, version=args.version, use_compile=args.compile, use_attn=not args.no_attn)
