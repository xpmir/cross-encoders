import torch
import torch.nn as nn
from unittest.mock import MagicMock
from xpm_torch.context import Context

from configuration import CE_FineTuning
from cross_encoders_training.experiment import get_ce_name
from training_utils import FreezeParameterHook


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.LayerNorm(16)
        self.mlp_norm = nn.LayerNorm(16)
        self.linear = nn.Linear(16, 16)
        self.custom_head = nn.Linear(16, 1)


def test_freeze_parameter_hook():
    model = DummyModel()
    context = MagicMock(spec=Context)
    context.model = model

    hook = FreezeParameterHook.C(pattern=r"(?i)norm")
    hook.before(context)

    assert not model.attn_norm.weight.requires_grad
    assert not model.attn_norm.bias.requires_grad
    assert not model.mlp_norm.weight.requires_grad
    assert not model.mlp_norm.bias.requires_grad
    assert model.linear.weight.requires_grad
    assert model.linear.bias.requires_grad
    assert model.custom_head.weight.requires_grad


def test_freeze_parameter_hook_custom_pattern():
    model = DummyModel()
    context = MagicMock(spec=Context)
    context.model = model

    hook = FreezeParameterHook.C(pattern=r"custom_head")
    hook.before(context)

    assert model.attn_norm.weight.requires_grad
    assert model.linear.weight.requires_grad
    assert not model.custom_head.weight.requires_grad
    assert not model.custom_head.bias.requires_grad


def test_freeze_ln_config_defaults():
    cfg = CE_FineTuning()
    assert cfg.freeze_LN is False
    assert cfg.freeze_LN_pattern == r"(?i)norm"


def test_get_ce_name_freeze_ln():
    tags = {"base": "bert-base-uncased", "freeze_LN": True}
    name = get_ce_name(tags)
    assert "freezeLN" in name


def test_freeze_parameter_hook_trainer_context(tmp_path):
    from xpm_torch.trainers.context import TrainerContext

    model = DummyModel()
    context = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=model,
        optimizer=None,
        fabric=None,
    )

    # Test that TrainerContext exposes .model property
    assert context.model is model

    hook = FreezeParameterHook.C(pattern=r"(?i)norm")
    hook.before(context)

    assert not model.attn_norm.weight.requires_grad
    assert not model.mlp_norm.weight.requires_grad
    assert model.linear.weight.requires_grad


def test_freeze_parameter_hook_state_model_fallback():
    model = DummyModel()

    class LegacyContext:
        def __init__(self, m):
            self.state = MagicMock()
            self.state.model = m

    context = LegacyContext(model)
    hook = FreezeParameterHook.C(pattern=r"(?i)norm")
    hook.before(context)

    assert not model.attn_norm.weight.requires_grad
    assert model.linear.weight.requires_grad


def test_freeze_parameter_hook_invalid_context_raises():
    import pytest

    class BadContext:
        pass

    hook = FreezeParameterHook.C(pattern=r"(?i)norm")
    with pytest.raises(AttributeError, match=r"Cannot resolve model from context"):
        hook.before(BadContext())
