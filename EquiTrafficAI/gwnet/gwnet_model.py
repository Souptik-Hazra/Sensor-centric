"""
EquiTraffic-GPT MLOps Module 1: State-of-the-Art Model Architecture Layer (gwnet_model.py)

SOTA PyTorch 2.x Implementation of Graph WaveNet (GWNet):
- Spatial Graph Convolution (GCN) with native einsum tensor contraction
- Gated Temporal Dilated Convolutions (TCN)
- Dynamic Spatial-Temporal Scaled Dot-Product Attention (F.scaled_dot_product_attention / FlashAttention)
- LayerNorm Residual & Skip Connection Stabilization
- Adaptive Spatial Adjacency Matrix Factorization (@ operator) with Xavier Uniform initialization
- Optional PyTorch 2.0+ torch.compile() Kernel Fusion Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class linear(nn.Module):
    """1x1 2D Convolutional Feature Projection Layer."""
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class SpatialTemporalAttention(nn.Module):
    """
    SOTA Dynamic Spatial-Temporal Attention using PyTorch 2.0 FlashAttention.
    Dynamically adjusts edge weights between non-adjacent highway nodes based on real-time speed drops.
    """
    def __init__(self, channels, num_heads=4):
        super(SpatialTemporalAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x shape: (B, C, N, L)
        b, c, n, l = x.shape
        if c % self.num_heads != 0:
            return x

        # Reshape to (B * L, N, C)
        x_perm = x.permute(0, 3, 2, 1).reshape(b * l, n, c)
        
        q = self.q_proj(x_perm).view(b * l, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_perm).view(b * l, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_perm).view(b * l, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention (PyTorch 2.0 FlashAttention C++ kernel)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(b * l, n, c)
        out = self.out_proj(attn_out).view(b, l, n, c).permute(0, 3, 2, 1)
        return x + out


class GCN(nn.Module):
    """Graph Convolutional Layer with Multi-Order Chebyshev Adjacency Support."""
    def __init__(self, c_in, c_out, dropout=0.3, support_len=3, order=2):
        super(GCN, self).__init__()
        self.order = order
        c_in_total = (order * support_len + 1) * c_in
        self.mlp = linear(c_in_total, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = torch.einsum('ncvl,vw->ncwl', x, a).contiguous()
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,vw->ncwl', x1, a).contiguous()
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return self.dropout(h)


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet (GWNet) SOTA 2026 Spatial-Temporal Architecture.
    Reference: Wu et al. (IJCAI 2019) + SOTA PyTorch 2.x Upgrades
    """
    def __init__(self, num_nodes, in_dim=3, out_dim=1, horizon=12, supports=None, adp_adj=True, 
                 dropout=0.3, residual_channels=32, dilation_channels=32, skip_channels=256, 
                 end_channels=512, kernel_size=2, blocks=4, layers=2, use_attn=True):
        super(GraphWaveNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.supports = supports if supports is not None else []
        self.supports_len = len(self.supports)
        self.adp_adj = adp_adj
        self.use_attn = use_attn

        if adp_adj:
            # Xavier Uniform initialization for smooth initial adaptive graph factorization
            self.nodevec1 = nn.Parameter(torch.empty(num_nodes, 10))
            self.nodevec2 = nn.Parameter(torch.empty(10, num_nodes))
            nn.init.xavier_uniform_(self.nodevec1)
            nn.init.xavier_uniform_(self.nodevec2)
            self.supports_len += 1

        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()

        if use_attn:
            self.spatial_attn = SpatialTemporalAttention(residual_channels, num_heads=4)

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=1)

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=1))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim * horizon, kernel_size=1)

    def forward(self, input):
        if input.dim() == 4 and input.size(1) != self.in_dim:
            input = input.transpose(1, 3)

        in_len = input.size(3)
        x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0)) if in_len < self.receptive_field else input

        # Device alignment safety for supports
        device_supports = [sup.to(input.device) for sup in self.supports]

        if self.adp_adj:
            adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
            new_supports = device_supports + [adp]
        else:
            new_supports = device_supports

        x = self.start_conv(x)
        if self.use_attn:
            x = self.spatial_attn(x)

        skip = torch.tensor(0.0, device=x.device)

        for i in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate_out = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate_out

            s = self.skip_convs[i](x)
            skip = skip[:, :, :, -s.size(3):] + s if skip.dim() == 4 else s

            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            
            # LayerNorm across residual channels & nodes
            b_sz, c_sz, n_sz, l_sz = x.shape
            x_ln = x.permute(0, 3, 1, 2)
            x_ln = self.ln[i](x_ln)
            x = x_ln.permute(0, 2, 3, 1)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        b, _, n, _ = x.shape
        return x.view(b, self.horizon, self.out_dim, n).transpose(2, 3)


if __name__ == "__main__":
    print("Testing SOTA PyTorch 2.x GWNet Architecture...")
    model = GraphWaveNet(num_nodes=207, in_dim=3, out_dim=1, horizon=12)
    dummy_x = torch.randn(2, 3, 207, 12)
    out = model(dummy_x)
    print(f"[+] SOTA GWNet Forward Pass Success: Input {dummy_x.shape} -> Output {out.shape}")
