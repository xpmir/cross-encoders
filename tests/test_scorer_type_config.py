"""Unit tests for scorer_type configuration and factory dispatch."""

import pytest
from configuration import CE_FineTuning, ScorerType
from cross_encoders_training.experiment import (
    build_STCrossEncoder,
    build_STMultiVectorEncoder,
    get_ce_name,
)
from experimaestro.experiments.grid import generate_grid
from xpmir.neural.sentence_transformers import STCrossEncoder, STMultiVectorEncoder


def test_scorer_type_enum():
    assert ScorerType.stCrossEncoder.value == "stCrossEncoder"
    assert ScorerType.stMultiVectorEncoder.value == "stMultiVectorEncoder"
    assert ScorerType.hfCrossScorer.value == "hfCrossScorer"


def test_ce_finetuning_default_scorer_type():
    cfg = CE_FineTuning()
    assert cfg.scorer_type == ScorerType.stCrossEncoder.value


def test_build_st_cross_encoder():
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L6-v2", max_length=128)
    scorer, init_tasks = build_STCrossEncoder(cfg)
    assert isinstance(scorer, STCrossEncoder.XPMConfig)
    assert len(init_tasks) == 1


def test_build_st_multivector_encoder():
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=128)
    scorer, init_tasks = build_STMultiVectorEncoder(cfg)
    assert isinstance(scorer, STMultiVectorEncoder.XPMConfig)
    assert len(init_tasks) == 0
    assert scorer.query_maxlen == 32
    assert scorer.doc_maxlen == 128


def test_get_ce_name_prefix():
    tags_ce = {"base": "bert-base", "loss": "bce", "scorer_type": "stCrossEncoder"}
    tags_mv = {"base": "bert-base", "loss": "bce", "scorer_type": "stMultiVectorEncoder"}
    tags_default = {"base": "bert-base", "loss": "bce"}

    assert get_ce_name(tags_ce).startswith("cross-encoder-")
    assert get_ce_name(tags_mv).startswith("colbert-")
    assert get_ce_name(tags_default).startswith("cross-encoder-")


def test_invalid_scorer_type_raises():
    with pytest.raises(ValueError):
        ScorerType("invalid_scorer")


def test_grid_search_scorer_type():
    from experimaestro.experiments.grid import GenericParams

    # Inline GenericParams
    cfg = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
        scorer_type=GenericParams(values_list=["stCrossEncoder", "stMultiVectorEncoder"]),
    )
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    types_in_configs = {c.scorer_type for c in configs}
    assert types_in_configs == {"stCrossEncoder", "stMultiVectorEncoder"}

    # Via grid_search dict
    cfg_dict = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
        grid_search={"scorer_type": ["stCrossEncoder", "stMultiVectorEncoder"]},
    )
    configs_dict, tags_dict = generate_grid(cfg_dict)
    assert len(configs_dict) == 2
    types_in_configs_dict = {c.scorer_type for c in configs_dict}
    assert types_in_configs_dict == {"stCrossEncoder", "stMultiVectorEncoder"}


def test_ettin_colbert_optimizer_regex():
    import torch.nn as nn
    from xpm_torch.experiments.configuration import TransformerOptimization
    from xpm_torch.optim import ScheduledOptimizer

    class MockEttinMultiVector(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleDict({
                "tok_embeddings": nn.Embedding(10, 16),
                "norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
            })
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn_norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
                    "Wqkv": nn.Linear(16, 48, bias=False),
                    "mlp_norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
                    "Wi": nn.Linear(16, 64, bias=False),
                })
            ])
            self.final_norm = nn.LayerNorm(16, elementwise_affine=True, bias=False)
            self.linear = nn.Linear(16, 16, bias=False)

    model = MockEttinMultiVector()
    opt_config = TransformerOptimization(re_no_l2_regularization=[r"\.bias$", r"(?i)norm"])
    optimizers = opt_config.optimizer

    sched_opt = ScheduledOptimizer()
    sched_opt.initialize(
        param_optimizers=optimizers,
        num_training_steps=100,
        module=model,
        use_scaler=False,
    )
    assert len(sched_opt.optimizers) == 2


def test_scheduled_optimizer_ignored_params_logging(caplog):
    import logging
    import torch.nn as nn
    from xpm_torch.experiments.configuration import TransformerOptimization
    from xpm_torch.optim import ScheduledOptimizer, RegexParameterFilter, ParameterOptimizer

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trained = nn.Linear(4, 4)
            self.ignored = nn.Linear(4, 4)

    model = MockModel()
    param_opts = [
        ParameterOptimizer.C(
            optimizer=TransformerOptimization().get_optimizer(True),
            filter=RegexParameterFilter.C(excludes=[r"ignored"]),
        )
    ]
    sched = ScheduledOptimizer()
    with caplog.at_level(logging.INFO):
        sched.initialize(param_opts, 10, model, False)

    # 1. Ignored parameters must be logged as WARNING
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Ignored parameters (2) NOT being optimized" in r.message for r in warning_records)
    assert any("ignored.weight" in r.message for r in warning_records)

    # 2. Optimized group info log must NOT dump individual parameter names
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    opt_info = [r for r in info_records if "Optimizing with" in r.message]
    assert len(opt_info) > 0
    for r in opt_info:
        assert "parameters)" in r.message
        assert "trained.weight" not in r.message  # No parameter names dumped at INFO

    # 3. When no parameters are ignored, INFO logs 'Ignored parameters: none'
    caplog.clear()
    all_param_opts = [
        ParameterOptimizer.C(
            optimizer=TransformerOptimization().get_optimizer(True),
        )
    ]
    with caplog.at_level(logging.INFO):
        sched.initialize(all_param_opts, 10, model, False)

    assert any("Ignored parameters: none" in r.message for r in caplog.records)
