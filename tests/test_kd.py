"""Unit test for Knowledge Distillation loss and models."""

import torch

from models.action_bilstm import ActionBiLSTMAttentionClassifier
from models.kd_loss import KnowledgeDistillationLoss


def test_kd_loss_computation():
    batch_size = 4
    num_classes = 15
    feat_dim = 256

    criterion = KnowledgeDistillationLoss(
        temperature=3.0, alpha_ce=0.4, beta_kd=0.4, gamma_feat=0.2, feat_loss_type="mse"
    )

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 2, 5, 14], dtype=torch.long)
    student_feats = torch.randn(batch_size, feat_dim, requires_grad=True)
    teacher_feats = torch.randn(batch_size, feat_dim)

    loss, metrics = criterion(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=targets,
        student_features=student_feats,
        teacher_features=teacher_feats,
    )

    assert torch.is_tensor(loss)
    assert loss.item() > 0.0
    assert "loss_total" in metrics
    assert "loss_ce" in metrics
    assert "loss_kd" in metrics
    assert "loss_feat" in metrics

    # Test backward pass
    loss.backward()
    assert student_logits.grad is not None
    assert student_feats.grad is not None


def test_model_return_features():
    model = ActionBiLSTMAttentionClassifier(
        input_size=24, hidden_size=128, num_layers=2, num_classes=15
    )
    x = torch.randn(2, 10, 24)
    lengths = torch.tensor([10, 8])

    logits, features = model(x, lengths, return_features=True)
    assert logits.shape == (2, 15)
    assert features.shape == (2, 256)
