from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=dim, eps=eps)
    b = F.normalize(b, p=2, dim=dim, eps=eps)
    return (a * b).sum(dim=dim)


def seq_mask_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
    device = lengths.device
    range_t = torch.arange(T, device=device).unsqueeze(0)  
    return range_t < lengths.unsqueeze(1)                  


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
    if mask_bt is None:
        return F.mse_loss(a, b)
    m = mask_bt.unsqueeze(-1).float()
    diff2 = (a - b) ** 2 * m
    denom = m.sum().clamp_min(1.0)
    return diff2.sum() / denom


@dataclass
class MBPReactionConfig:
    d_model: int = 256
    theta: float = 0.2           
    use_soft_mask: bool = True  
    temperature: float = 0.1    
    straight_through: bool = True
    enable_lip: bool = True

class MBPReaction(nn.Module):
    def __init__(self, cfg: MBPReactionConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.proj_a = nn.LazyLinear(D)
        self.proj_f = nn.LazyLinear(D)
        self.proj_lip = nn.LazyLinear(D) if cfg.enable_lip else None
        self.proj_l = nn.LazyLinear(D)   
        self.ln_a = nn.LayerNorm(D)
        self.ln_f = nn.LayerNorm(D)
        self.ln_lip = nn.LayerNorm(D) if cfg.enable_lip else None
        self.ln_l = nn.LayerNorm(D)

    @staticmethod
    def _make_mask(sim: torch.Tensor, theta: float, use_soft: bool, tau: float, straight_through: bool, training: bool) -> torch.Tensor:
        if use_soft:
            m = torch.sigmoid((sim - theta) / max(tau, 1e-6))
            if training and straight_through:
                hard = (m > 0.5).float()
                m = hard.detach() + (m - m.detach())
            return m
        else:
            return (sim > theta).float()

    @staticmethod
    def _apply_mask(x: torch.Tensor, m_bt: torch.Tensor) -> torch.Tensor:
        return x * m_bt.unsqueeze(-1)

    def forward(
        self,
        *,
        speaker_audio: torch.Tensor,          
        speaker_3dmm: torch.Tensor,           
        speaker_audio2text: torch.Tensor,    
        speaker_lip: Optional[torch.Tensor] = None,  
        seq_lengths: Optional[torch.Tensor] = None,  
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, _ = speaker_audio.shape

        valid_bt = seq_mask_from_lengths(seq_lengths, T) if seq_lengths is not None else None

        A = self.ln_a(self.proj_a(speaker_audio))           
        Fm = self.ln_f(self.proj_f(speaker_3dmm))           
        L = self.ln_l(self.proj_l(speaker_audio2text))      
        if cfg.enable_lip and speaker_lip is not None:
            LP = self.ln_lip(self.proj_lip(speaker_lip))    
        else:
            LP = None

        sim_a = cosine_sim(A, L, dim=-1)                    
        sim_f = cosine_sim(Fm, L, dim=-1)                  
        sim_lip = cosine_sim(LP, L, dim=-1) if LP is not None else None

        if valid_bt is not None:
            sim_a = sim_a.masked_fill(~valid_bt, 0.0)
            sim_f = sim_f.masked_fill(~valid_bt, 0.0)
            if sim_lip is not None:
                sim_lip = sim_lip.masked_fill(~valid_bt, 0.0)

        mask_a = self._make_mask(sim_a, cfg.theta, cfg.use_soft_mask, cfg.temperature, cfg.straight_through, self.training)
        mask_f = self._make_mask(sim_f, cfg.theta, cfg.use_soft_mask, cfg.temperature, cfg.straight_through, self.training)
        mask_lip = self._make_mask(sim_lip, cfg.theta, cfg.use_soft_mask, cfg.temperature, cfg.straight_through, self.training) if sim_lip is not None else None

        A_hat = self._apply_mask(A, mask_a)
        F_hat = self._apply_mask(Fm, mask_f)
        LP_hat = self._apply_mask(LP, mask_lip) if LP is not None else None

        L_a = masked_mse(A, A_hat, valid_bt)
        L_f = masked_mse(Fm, F_hat, valid_bt)
        L_lip = masked_mse(LP, LP_hat, valid_bt) if LP is not None else torch.tensor(0.0, device=A.device)
        L_r_m = L_a + L_f + L_lip

        return {
            "A_proj": A,            
            "F_proj": Fm,           
            "L_proj": L,            
            "LIP_proj": LP,         
            "A_hat": A_hat,         
            "F_hat": F_hat,
            "LIP_hat": LP_hat,
            "mask_a": mask_a,       
            "mask_f": mask_f,
            "mask_lip": mask_lip,   # or None
            "sim_a": sim_a,
            "sim_f": sim_f,
            "sim_lip": sim_lip,     
            "L_r_m": L_r_m,
            "valid_mask": valid_bt, # or None
        }

