"""
Module: kd_loss.py
Purpose: Knowledge Distillation loss module for skeleton-based action recognition.
         Inspired by Liu et al. (Signal Processing 2024):
         Multi-level distillation with:
         1. Hard task classification loss (Cross-Entropy)
         2. Soft logit distillation loss (KL Divergence with Temperature)
         3. Latent feature representation distillation (MSE / Cosine Distance)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss for Knowledge Distillation:
        L_total = alpha * L_CE(z_S, y) + beta * L_KD(z_S, z_T; T) + gamma * L_feat(h_S, h_T)
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha_ce: float = 0.4,
        beta_kd: float = 0.4,
        gamma_feat: float = 0.2,
        feat_loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.gamma_feat = gamma_feat
        self.feat_loss_type = feat_loss_type

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the composite KD loss.

        Args:
            student_logits: [Batch, NumClasses] raw logits from student
            teacher_logits: [Batch, NumClasses] raw logits from teacher (detached)
            targets: [Batch] ground-truth class labels
            student_features: [Batch, FeatureDim] intermediate spatiotemporal feature from student
            teacher_features: [Batch, FeatureDim] intermediate spatiotemporal feature from teacher
        """
        # 1. Hard Cross-Entropy Loss
        l_ce = self.ce_loss(student_logits, targets)

        # 2. Soft Logit Distillation Loss (KL Divergence)
        # Scaled by temperature^2 as standard in Hinton's KD
        p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        l_kd = self.kl_loss(p_s, p_t) * (self.temperature**2)

        # 3. Intermediate Feature Representation Distillation
        l_feat = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            if self.feat_loss_type == "cosine":
                # 1 - cosine_similarity
                l_feat = (
                    1.0 - F.cosine_similarity(student_features, teacher_features, dim=1)
                ).mean()
            else:
                l_feat = self.mse_loss(student_features, teacher_features)

        # Total weighted loss
        total_loss = self.alpha_ce * l_ce + self.beta_kd * l_kd + self.gamma_feat * l_feat

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(l_ce.item()),
            "loss_kd": float(l_kd.item()),
            "loss_feat": float(l_feat.item()) if isinstance(l_feat, torch.Tensor) else 0.0,
        }

        return total_loss, metrics
