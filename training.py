from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def lengths_to_mask(lengths: Optional[torch.Tensor], T: int, device: torch.device) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    ar = torch.arange(T, device=device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)  # (B,T) bool


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
    if mask_bt is None:
        return F.mse_loss(a, b)
    m = mask_bt.unsqueeze(-1).float()
    diff2 = (a - b) ** 2 * m
    denom = m.sum().clamp_min(1.0)
    return diff2.sum() / denom


def time_mean_masked(x_bt: torch.Tensor, mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
    if mask_bt is None:
        return x_bt.mean(dim=1)
    m = mask_bt.float()
    s = (x_bt * m).sum(dim=1)
    d = m.sum(dim=1).clamp_min(1.0)
    return s / d

def loss_mbp_reconstruction(A: torch.Tensor, A_hat: torch.Tensor,
                             F: torch.Tensor, F_hat: torch.Tensor,
                             lengths: Optional[torch.Tensor] = None) -> torch.Tensor:

    device = A.device
    T = A.shape[1]
    mask = lengths_to_mask(lengths, T, device) if lengths is not None else None
    La = masked_mse(A, A_hat, mask)
    Lf = masked_mse(F, F_hat, mask)
    return La + Lf

def loss_fine_disentanglement(Z_f: torch.Tensor,
                              lengths: Optional[torch.Tensor] = None,
                              tau: float = 0.07) -> torch.Tensor:
    if Z_f.dim() == 3:
        Z_f = Z_f.squeeze(-1)
    B, T = Z_f.shape
    device = Z_f.device
    mask = lengths_to_mask(lengths, T, device) if lengths is not None else None
    s = time_mean_masked(Z_f, mask)  
    logits = s.unsqueeze(1).expand(B, B)  
    targets = torch.arange(B, device=device)
    return F.cross_entropy(logits / max(tau, 1e-6), targets)


def loss_coarse_disentanglement(Z_c: torch.Tensor,
                                lengths: Optional[torch.Tensor] = None,
                                tau: float = 0.07) -> torch.Tensor:
    if Z_c.dim() == 3:
        Z_c = Z_c.squeeze(-1)
    B, T = Z_c.shape
    device = Z_c.device
    mask = lengths_to_mask(lengths, T, device) if lengths is not None else None
    s = -time_mean_masked(Z_c, mask)  
    logits = s.unsqueeze(1).expand(B, B)
    targets = torch.arange(B, device=device)
    return F.cross_entropy(logits / max(tau, 1e-6), targets)


def loss_motion_smooth(P: torch.Tensor,
                       lengths: Optional[torch.Tensor] = None,
                       w: int = 1) -> torch.Tensor:
    if w < 1:
        w = 1
    if lengths is None:
        P_prev = F.pad(P, (0, 0, 0, 0, w, 0), mode="replicate")[:, :-w]
        P_next = F.pad(P, (0, 0, 0, 0, 0, w), mode="replicate")[:, w:]
        diff = P - 0.5 * (P_prev + P_next)
        return (diff ** 2).mean()
    B, T, D = P.shape
    device = P.device
    mask = lengths_to_mask(lengths, T, device)
    valid = mask.clone()
    valid = valid & torch.roll(mask, 1, dims=1) & torch.roll(mask, -1, dims=1)
    m = valid.unsqueeze(-1).float()
    P_prev = torch.roll(P, shifts= w, dims=1)
    P_next = torch.roll(P, shifts=-w, dims=1)
    diff = (P - 0.5 * (P_prev + P_next)) * m
    denom = m.sum().clamp_min(1.0)
    return (diff ** 2).sum() / denom





