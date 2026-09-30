"""
Unit tests for evaluation grid search and from_experiment resolution.
Run with: `uv run pytest tests/test_evaluation_grid_and_experiment.py -v`
"""

import pytest
from experimaestro.experiments.grid import generate_grid

from configuration import ScorerType
from Evaluation.experiment import (
    BaselinesConfig,
    resolve_evaluation_scorer,
    evaluate_scorer,
)


def test_scorer_type_from_string_resolution():
    """Test resolution of various string representations to ScorerType enum members."""
    # HF Cross Scorer aliases
    assert ScorerType.from_string("hf") == ScorerType.hfCrossScorer
    assert ScorerType.from_string("hf_cross_scorer") == ScorerType.hfCrossScorer
    assert ScorerType.from_string("hfcrossscorer") == ScorerType.hfCrossScorer
    assert ScorerType.from_string("HF_CROSS_SCORER") == ScorerType.hfCrossScorer

    # Sentence Transformers aliases
    assert ScorerType.from_string("st") == ScorerType.stCrossEncoder
    assert ScorerType.from_string("sentence_transformers") == ScorerType.stCrossEncoder
    assert ScorerType.from_string("sentencetransformers") == ScorerType.stCrossEncoder

    # ST MultiVector
    assert ScorerType.from_string("st_multivector") == ScorerType.stMultiVectorEncoder
    assert ScorerType.from_string("st_multivector_encoder") == ScorerType.stMultiVectorEncoder
    assert ScorerType.from_string("multivector") == ScorerType.stMultiVectorEncoder

    # MICE v2
    assert ScorerType.from_string("micev2") == ScorerType.miceV2
    assert ScorerType.from_string("mice_v2") == ScorerType.miceV2
    assert ScorerType.from_string("MICEV2") == ScorerType.miceV2

    # from_experiment and auto
    assert ScorerType.from_string("from_experiment") == ScorerType.from_experiment
    assert ScorerType.from_string("auto") == ScorerType.auto

    # Enum instance pass-through
    assert ScorerType.from_string(ScorerType.miceV2) == ScorerType.miceV2

    # Invalid string raises ValueError
    with pytest.raises(ValueError, match="Unknown scorer_type"):
        ScorerType.from_string("invalid_scorer_type_xyz")


def test_baselines_config_single_config():
    """Test BaselinesConfig generate_grid with single parameters."""
    cfg = BaselinesConfig(
        scorer_id="cross-encoder/ms-marco-MiniLM-L12-v2",
        scorer_type=ScorerType.hfCrossScorer,
        max_length=256,
    )
    all_configs, all_tags = generate_grid(cfg)
    assert len(all_configs) == 1
    assert all_configs[0].scorer_id == "cross-encoder/ms-marco-MiniLM-L12-v2"
    assert all_configs[0].scorer_type == ScorerType.hfCrossScorer


def test_baselines_config_grid_search_scorer_id():
    """Test grid_search over scorer_id."""
    cfg = BaselinesConfig(
        scorer_type="hf",
        grid_search={
            "scorer_id": [
                "cross-encoder/ms-marco-MiniLM-L12-v2",
                "xpmir/cross-encoder-MiniLM-L12-MarginMSE",
            ]
        },
    )
    all_configs, all_tags = generate_grid(cfg)
    assert len(all_configs) == 2
    scorer_ids = [c.scorer_id for c in all_configs]
    assert "cross-encoder/ms-marco-MiniLM-L12-v2" in scorer_ids
    assert "xpmir/cross-encoder-MiniLM-L12-MarginMSE" in scorer_ids

    # Check tags extracted
    assert all_tags[0].get("scorer_id") == all_configs[0].scorer_id
    assert all_tags[1].get("scorer_id") == all_configs[1].scorer_id


def test_baselines_config_cartesian_grid_search():
    """Test cartesian product of scorer_id and hard_prune_threshold."""
    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        grid_search={
            "hard_prune_threshold": [0.0, 0.2, 0.5],
        },
    )
    all_configs, all_tags = generate_grid(cfg)
    assert len(all_configs) == 3
    thresholds = [c.hard_prune_threshold for c in all_configs]
    assert thresholds == [0.0, 0.2, 0.5]
    for tag_map in all_tags:
        assert "hard_prune_threshold" in tag_map


def test_resolve_evaluation_scorer_from_experiment_hub():
    """Test resolving a MICE v2 model from Hugging Face Hub using from_experiment."""
    from MICE.modeling.mice_v2.modernbert_xpm import ModernBertMiceScorer

    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        max_query_length=24,
        max_doc_length=180,
        hard_prune_threshold=0.35,
        hard_pruning=True,
    )
    scorer, init_tasks, model_loader, resolved_type_str = resolve_evaluation_scorer(
        cfg.from_experiment, cfg
    )

    assert isinstance(scorer, ModernBertMiceScorer)
    assert scorer.tokenizer.max_query_length == 24
    assert scorer.tokenizer.max_doc_length == 180
    assert scorer.hard_prune_threshold == 0.35
    assert scorer.hard_pruning is True
    assert resolved_type_str == "mice_v2"
    assert len(init_tasks) >= 1


def test_resolve_evaluation_scorer_eval_sparsity():
    """Test that eval_sparsity appends InitEvalSparsityHook."""
    from MICE.modeling.mice_v2.modernbert_xpm import InitEvalSparsityHook

    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        eval_sparsity=True,
        hard_pruning=True,
        hard_prune_threshold=0.4,
    )
    scorer, init_tasks, model_loader, resolved_type_str = resolve_evaluation_scorer(
        cfg.from_experiment, cfg
    )
    sparsity_hooks = [t for t in init_tasks if isinstance(t, InitEvalSparsityHook)]
    assert len(sparsity_hooks) == 1
    assert sparsity_hooks[0].threshold == 0.4


def test_evaluate_scorer_missing_scorer_raises():
    """Test that missing experiment/scorer reference in evaluate_scorer raises ValueError."""
    cfg = BaselinesConfig()
    with pytest.raises(ValueError, match="Neither 'scorer_id' nor 'from_experiment'"):
        evaluate_scorer(cfg, {}, None, {}, None, None, None)


def test_lateon_plaid_nbits_grid_generation():
    """Test grid generation for LateOn + MICE multi-configuration evaluation."""
    cfg = BaselinesConfig(
        use_plaid=True,
        retriever="naver/splade-v3-distilbert",
        grid_search={
            "plaid.n_bits": [2, 4, 8],
            "config_dicts": [
                {
                    "scorer_type": "stMultiVectorEncoder",
                    "scorer_id": "lightonai/LateOn",
                },
                {
                    "scorer_type": "miceV2",
                    "scorer_id": "train_micev2_ettin68m_l10+4_v3_plaid_128",
                },
            ],
        },
    )
    all_configs, all_tags = generate_grid(cfg)
    assert len(all_configs) == 6

    # 3 LateOn configs with PLAID (ColBERT + PLAID)
    lateon_cfgs = [c for c in all_configs if c.scorer_id == "lightonai/LateOn"]
    assert len(lateon_cfgs) == 3
    assert {c.plaid.n_bits for c in lateon_cfgs} == {2, 4, 8}
    assert all(c.use_plaid for c in lateon_cfgs)

    # 3 MICE configs with PLAID
    mice_cfgs = [
        c
        for c in all_configs
        if c.scorer_id == "train_micev2_ettin68m_l10+4_v3_plaid_128"
    ]
    assert len(mice_cfgs) == 3
    assert {c.plaid.n_bits for c in mice_cfgs} == {2, 4, 8}
    assert all(c.retriever == "naver/splade-v3-distilbert" for c in mice_cfgs)


def test_mice_plaid_nbits_grid_generation():
    """Test grid generation for MICE across multiple bit-rates."""
    cfg = BaselinesConfig(
        use_plaid=True,
        retriever="naver/splade-v3-distilbert",
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        grid_search={"plaid.n_bits": [2, 4, 8, 16]},
    )
    all_configs, all_tags = generate_grid(cfg)
    assert len(all_configs) == 4
    assert [c.plaid.n_bits for c in all_configs] == [2, 4, 8, 16]
    assert all(c.retriever == "naver/splade-v3-distilbert" for c in all_configs)


def test_evaluate_scorer_colbert_plaid_dispatch():
    """Test that ColBERT + PLAID directly dispatches PlaidRetriever without needing first_stage."""
    from unittest.mock import MagicMock

    mock_tests = MagicMock()
    cfg = BaselinesConfig(
        scorer_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer_type="stMultiVectorEncoder",
        use_plaid=True,
    )
    # ColBERT + PLAID requires no first_stage_factories
    evaluate_scorer(
        cfg=cfg,
        cfg_tags={"n_bits": 2},
        tests=mock_tests,
        first_stage_factories={},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )
    assert mock_tests.evaluate_retriever.call_count == 1
    call_args = mock_tests.evaluate_retriever.call_args
    # First argument is a partial of plaid_retriever
    assert call_args[0][0].func.__name__ == "plaid_retriever"


def test_evaluate_scorer_colbert_without_plaid_warning(caplog):
    """Test that ColBERT without PLAID logs a loud warning and sets plaid_retriever=False."""
    import logging
    from unittest.mock import MagicMock

    mock_tests = MagicMock()
    mock_fs_factory = MagicMock()
    cfg = BaselinesConfig(
        scorer_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer_type="stMultiVectorEncoder",
        use_plaid=False,
    )
    with caplog.at_level(logging.WARNING):
        evaluate_scorer(
            cfg=cfg,
            cfg_tags={},
            tests=mock_tests,
            first_stage_factories={"splade": (mock_fs_factory, [])},
            launcher_evaluate=None,
            launcher_index=None,
            fabric_config=None,
        )
    assert any(
        "Evaluating ColBERT model" in record.message
        and "second-stage reranker" in record.message
        for record in caplog.records
    )
    assert mock_tests.evaluate_retriever.call_count == 1
    call_args = mock_tests.evaluate_retriever.call_args
    retriever_partial = call_args[0][0]
    assert retriever_partial.keywords["scorer"].tags()["plaid_retriever"] is False


def test_evaluate_scorer_missing_first_stage_raises():
    """Test that evaluate_scorer raises ValueError if a reranker lacks first_stage_factories."""
    from unittest.mock import MagicMock

    mock_tests = MagicMock()
    cfg = BaselinesConfig(
        scorer_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer_type="stCrossEncoder",
        use_plaid=False,
    )
    with pytest.raises(ValueError, match="requires a first-stage candidate retriever"):
        evaluate_scorer(
            cfg=cfg,
            cfg_tags={},
            tests=mock_tests,
            first_stage_factories={},
            launcher_evaluate=None,
            launcher_index=None,
            fabric_config=None,
        )
