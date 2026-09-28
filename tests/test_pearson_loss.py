"""Unit tests for Pearson correlation loss (PearsonLoss) and its integration."""

import pytest
import torch
import numpy as np

from configuration import Losses, CE_FineTuning, ScorerType
from training_utils import PearsonLoss, build_trainer
from cross_encoders_training.experiment import build_STMultiVectorEncoder
from pointwise_distillation import PointwiseDistillationTrainer
from datamaestro_ir.data.huggingface import HuggingFacePointwiseDistillationSamples


def test_pearson_loss_mathematical_bounds():
    """Verify that PearsonLoss accurately computes 1 - r for known correlations."""
    loss_cfg = PearsonLoss.C(eps=1e-8)
    loss_fn = loss_cfg.instance()
    loss_fn.initialize(None)

    # 1. Perfect positive correlation (s = 3 * y + 5) -> r = 1.0, loss = 0.0
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    s_pos = 3.0 * y + 5.0
    loss_pos = loss_fn.compute(s_pos, y)
    assert pytest.approx(loss_pos.item(), abs=1e-5) == 0.0

    # 2. Perfect negative correlation (s = -2 * y + 1) -> r = -1.0, loss = 2.0
    s_neg = -2.0 * y + 1.0
    loss_neg = loss_fn.compute(s_neg, y)
    assert pytest.approx(loss_neg.item(), abs=1e-5) == 2.0

    # 3. Uncorrelated / orthogonal vectors -> r = 0.0, loss = 1.0
    y_uncorr = torch.tensor([1.0, -1.0, 1.0, -1.0])
    s_uncorr = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss_uncorr = loss_fn.compute(s_uncorr, y_uncorr)
    assert pytest.approx(loss_uncorr.item(), abs=1e-5) == 1.0


def test_pearson_loss_collapse_prevention():
    """Verify that uniform student predictions (collapse) do not cause NaN/Inf and yield finite gradients."""
    eps = 1e-6
    loss_cfg = PearsonLoss.C(eps=eps)
    loss_fn = loss_cfg.instance()
    loss_fn.initialize(None)

    # Case A: Student has zero variance (all predictions equal)
    s_collapsed = torch.full((32,), 0.85, requires_grad=True)
    y_teacher = torch.randn(32)

    loss_collapsed = loss_fn.compute(s_collapsed, y_teacher)
    assert not torch.isnan(loss_collapsed)
    assert not torch.isinf(loss_collapsed)
    assert pytest.approx(loss_collapsed.item(), abs=1e-4) == 1.0

    # Backward pass must generate finite, non-NaN gradients
    loss_collapsed.backward()
    assert s_collapsed.grad is not None
    assert not torch.isnan(s_collapsed.grad).any(), "NaN found in gradients during student collapse!"
    assert not torch.isinf(s_collapsed.grad).any(), "Inf found in gradients during student collapse!"
    assert not (s_collapsed.grad == 0).all(), "Gradients should be non-zero to pull student out of collapse!"

    # Case B: Teacher has zero variance
    s_rand = torch.randn(32, requires_grad=True)
    y_constant = torch.full((32,), 10.0)
    loss_teacher_const = loss_fn.compute(s_rand, y_constant)
    assert not torch.isnan(loss_teacher_const)
    assert pytest.approx(loss_teacher_const.item(), abs=1e-4) == 1.0

    loss_teacher_const.backward()
    assert not torch.isnan(s_rand.grad).any()


def test_pearson_loss_tensor_shapes():
    """Verify that PearsonLoss handles (B,), (B, 1), and multidimensional inputs cleanly."""
    loss_cfg = PearsonLoss.C(eps=1e-6)
    loss_fn = loss_cfg.instance()
    loss_fn.initialize(None)

    # 1D inputs
    s_1d = torch.randn(16)
    y_1d = torch.randn(16)
    loss_1d = loss_fn.compute(s_1d, y_1d)

    # 2D (B, 1) inputs
    s_2d = s_1d.unsqueeze(-1)
    y_2d = y_1d.unsqueeze(-1)
    loss_2d = loss_fn.compute(s_2d, y_2d)

    assert pytest.approx(loss_1d.item(), rel=1e-6) == loss_2d.item()


def test_pearson_loss_custom_epsilon():
    """Verify that custom epsilon values are properly passed and used."""
    custom_eps = 1e-4
    loss_cfg = PearsonLoss.C(eps=custom_eps)
    loss_fn = loss_cfg.instance()
    assert loss_fn.eps == custom_eps


def test_build_trainer_pearson_mixedbread_large(monkeypatch):
    """Verify that build_trainer correctly configures PearsonLoss for pearson_mixedbread_large."""
    import training_utils

    dummy_ds = HuggingFacePointwiseDistillationSamples.C(
        repo_id="dummy", split="train", score_field="label"
    )
    monkeypatch.setattr(
        training_utils, "build_ettin_reranker_v1_data", lambda *args, **kwargs: dummy_ds
    )

    # 1. Multi-vector encoder with pearson_mixedbread_large
    cfg_mv = CE_FineTuning(
        scorer_type=ScorerType.stMultiVectorEncoder.value,
        pearson_eps=1e-5,
    )
    cfg_mv.learner.loss = Losses.pearson_mixedbread_large.value
    trainer_mv = build_trainer(cfg_mv)

    assert isinstance(trainer_mv, PointwiseDistillationTrainer.XPMConfig)
    assert isinstance(trainer_mv.lossfn, PearsonLoss.XPMConfig)
    assert trainer_mv.lossfn.eps == 1e-5

    # 2. Cross encoder with pearson_mixedbread_large
    cfg_ce = CE_FineTuning(
        scorer_type=ScorerType.stCrossEncoder.value,
        pearson_eps=2e-6,
    )
    cfg_ce.learner.loss = Losses.pearson_mixedbread_large.value
    trainer_ce = build_trainer(cfg_ce)

    assert isinstance(trainer_ce, PointwiseDistillationTrainer.XPMConfig)
    assert isinstance(trainer_ce.lossfn, PearsonLoss.XPMConfig)
    assert trainer_ce.lossfn.eps == 2e-6


def test_similarity_fn_resolution_for_pearson():
    """Verify that build_STMultiVectorEncoder defaults to 'meanmaxsim' for pearson_mixedbread_large."""
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cfg.learner.loss = Losses.pearson_mixedbread_large.value
    scorer, _ = build_STMultiVectorEncoder(cfg)
    assert scorer.similarity_fn_name == "meanmaxsim"

    # Explicit override takes precedence
    cfg_override = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L-6-v2", similarity_fn_name="maxsim"
    )
    cfg_override.learner.loss = Losses.pearson_mixedbread_large.value
    scorer_override, _ = build_STMultiVectorEncoder(cfg_override)
    assert scorer_override.similarity_fn_name == "maxsim"
