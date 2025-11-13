from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, p=2, dim=dim, eps=eps)
    b_n = F.normalize(b, p=2, dim=dim, eps=eps)
    return (a_n * b_n).sum(dim=dim)


def gaussian_w2_diag(mu1: torch.Tensor, var1: torch.Tensor,
                     mu2: torch.Tensor, var2: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """W2 distance between diagonal Gaussians per frame. Returns W2, shape (B,T)."""
    var1 = var1.clamp_min(eps)
    var2 = var2.clamp_min(eps)
    term_mean = (mu1 - mu2).pow(2).sum(dim=-1)
    term_cov  = (var1.sqrt() - var2.sqrt()).pow(2).sum(dim=-1)
    return (term_mean + term_cov + eps).sqrt()


def lengths_to_mask(lengths: Optional[torch.Tensor], T: int, device: torch.device) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    ar = torch.arange(T, device=device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)  


def masked_time_mean(x_bt1: torch.Tensor, mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over time with boolean mask. x_bt1: (B,T,1) -> returns (B,)"""
    if mask_bt is None:
        return x_bt1.mean(dim=1).squeeze(-1)
    m = mask_bt.unsqueeze(-1).float()
    s = (x_bt1 * m).sum(dim=1)
    d = m.sum(dim=1).clamp_min(1.0)
    return (s / d).squeeze(-1)


@dataclass
class HCDConfig:
    d_model: int = 256
    d_fine: int = 64
    d_coarse: int = 128
    # Fine z_f = α( cos − β )
    alpha: float = 1.0
    beta: float = 0.0
    # Coarse z_c = γ( W2 − δ )
    gamma: float = 1.0
    delta: float = 0.0
    # InfoNCE temperature
    tau_fine: float = 0.07
    tau_coarse: float = 0.07
    # Local region heads
    num_regions: int = 3   # lip, eye, pose (placeholders)


class LocalRegions(nn.Module):
    """Learned local projectors approximating lip/eye/pose subsets.
    Given face F (B,T,D), produces fused local feature (B,T,d_fine).
    Replace with explicit landmark-region extractors if available.
    """
    def __init__(self, d_in: int, d_fine: int, num_regions: int = 3):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_fine), nn.ReLU(), nn.LayerNorm(d_fine))
            for _ in range(num_regions)
        ])
        self.gate = nn.Sequential(nn.Linear(d_in, num_regions), nn.Sigmoid())

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        regions = torch.stack([p(F) for p in self.projs], dim=2)  
        gates = self.gate(F).unsqueeze(-1)                        
        return (regions * gates).sum(dim=2)                       


class GlobalGaussianHead(nn.Module):
    """Produce per-frame (μ, σ^2) for diagonal Gaussian with light temporal context."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.depthwise = nn.Conv1d(d_in, d_in, kernel_size=3, padding=1, groups=d_in)
        self.norm = nn.LayerNorm(d_in)
        self.mu = nn.Linear(d_in, d_out)
        self.logvar = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        mu = self.mu(x)
        var = F.softplus(self.logvar(x))
        return mu, var


class HCD(nn.Module):
    def __init__(self, cfg: HCDConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.ln_a = nn.LayerNorm(D)
        self.ln_l = nn.LayerNorm(D)
        self.ln_f = nn.LayerNorm(D)
        self.local_face = LocalRegions(d_in=D, d_fine=cfg.d_fine, num_regions=cfg.num_regions)
        self.proj_a_f = nn.Linear(D, cfg.d_fine)
        self.ln_fine_f = nn.LayerNorm(cfg.d_fine)
        self.ln_fine_a = nn.LayerNorm(cfg.d_fine)
        self.face_head = GlobalGaussianHead(d_in=D, d_out=cfg.d_coarse)
        self.fuse_al = nn.Sequential(nn.Linear(2*D, 2*D), nn.ReLU(), nn.LayerNorm(2*D))
        self.al_head = GlobalGaussianHead(d_in=2*D, d_out=cfg.d_coarse)

    def forward(self, A: torch.Tensor, L: torch.Tensor, F: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, D = F.shape
        device = F.device
        mask_bt = lengths_to_mask(lengths, T, device)

        A = self.ln_a(A); L = self.ln_l(L); F = self.ln_f(F)

        F_loc = self.ln_fine_f(self.local_face(F))           
        A_loc = self.ln_fine_a(self.proj_a_f(A))              
        sim = cosine_sim(F_loc, A_loc, dim=-1)                
        Z_f = cfg.alpha * (sim - cfg.beta)
        if mask_bt is not None:
            Z_f = Z_f.masked_fill(~mask_bt, 0.0)
        Z_f = Z_f.unsqueeze(-1)                               

        AL = torch.cat([A, L], dim=-1)
        AL = self.fuse_al(AL)
        mu_f, var_f = self.face_head(F)                       
        mu_al, var_al = self.al_head(AL)                      
        w2 = gaussian_w2_diag(mu_f, var_f, mu_al, var_al)     
        Z_c = cfg.gamma * (w2 - cfg.delta)
        if mask_bt is not None:
            Z_c = Z_c.masked_fill(~mask_bt, 0.0)
        Z_c = Z_c.unsqueeze(-1)                              

        
        logits_f = masked_time_mean(Z_f, mask_bt).unsqueeze(1).expand(B, B)
        targets = torch.arange(B, device=device)
        L_fd = F.cross_entropy(logits_f / cfg.tau_fine, targets)
        logits_c = (-masked_time_mean(Z_c, mask_bt)).unsqueeze(1).expand(B, B)
        L_cd = F.cross_entropy(logits_c / cfg.tau_coarse, targets)

        return {
            "Z_f": Z_f,          
            "Z_c": Z_c,         
            "L_fd": L_fd,        
            "L_cd": L_cd,        
            "mu_f": mu_f, "var_f": var_f,
            "mu_al": mu_al, "var_al": var_al,
            "w2": w2,           
        }


