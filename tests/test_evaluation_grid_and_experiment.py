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
    assert retriever_partial.keywords["scorer"] is not None
    assert "plaid_retriever" not in retriever_partial.keywords["scorer"].tags()


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


def test_validate_and_align_config_with_baselines_config(caplog):
    """Test that validate_and_align_config aligns parameters from xpm_model_config with BaselinesConfig."""
    import logging
    from types import SimpleNamespace
    from training_utils import validate_and_align_config

    mock_tok = SimpleNamespace(max_query_length=32, max_doc_length=128)
    mock_model_cfg = SimpleNamespace(
        base="jhu-clsp/ettin-encoder-68m",
        dim_reduction=128,
        k_features=10,
        tokenizer=mock_tok,
    )

    # 1. Test inheritance when BaselinesConfig has defaults (None)
    cfg = BaselinesConfig(
        from_experiment="dummy/xp",
        scorer_type="miceV2",
    )
    tags = {}
    with caplog.at_level(logging.INFO):
        validate_and_align_config(
            xp_config=cfg,
            xpm_model_config=mock_model_cfg,
            tags=tags,
            from_ref="mock_xp",
        )

    assert cfg.max_query_length == 32
    assert cfg.max_doc_length == 128
    assert tags["base"] == "jhu-clsp/ettin-encoder-68m"
    assert any(
        "Inherited 'max_query_length' from loaded model" in r.message
        for r in caplog.records
    )

    # 2. Test explicit override
    caplog.clear()
    cfg_override = BaselinesConfig(
        from_experiment="dummy/xp",
        scorer_type="miceV2",
        max_query_length=48,  # Different from default None and model 32
    )
    with caplog.at_level(logging.WARNING):
        validate_and_align_config(
            xp_config=cfg_override,
            xpm_model_config=mock_model_cfg,
            tags=tags,
            from_ref="mock_xp",
        )

    assert cfg_override.max_query_length == 48
    assert any(
        "Config explicitly overrides loaded model parameter 'max_query_length'"
        in r.message
        for r in caplog.records
    )


def test_validate_scorer_init_tasks_catches_mismatch():
    """Test that validate_scorer_init_tasks raises ValueError when loader or hook points to a different scorer."""
    from experimaestro.core.objects.config import clone
    from Evaluation.experiment import validate_scorer_init_tasks
    from MICE.modeling.mice_v2.modernbert_xpm import InitEvalSparsityHook

    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
    )
    scorer, init_tasks, _, _ = resolve_evaluation_scorer(cfg.from_experiment, cfg)

    # 1. Matching scorer and init_tasks passes
    validate_scorer_init_tasks(scorer, init_tasks)

    # 2. Cloned (detached) scorer causes ValueError
    cloned_scorer = clone(scorer)
    with pytest.raises(
        ValueError, match="is bound to a different scorer instance"
    ):
        validate_scorer_init_tasks(cloned_scorer, init_tasks)

    # 3. Sparsity hook pointing to another scorer causes ValueError
    sparsity_hook = InitEvalSparsityHook.C(scorer=scorer, threshold=0.3)
    with pytest.raises(
        ValueError, match="hook points to a different scorer instance"
    ):
        validate_scorer_init_tasks(cloned_scorer, [sparsity_hook])


def test_evaluate_scorer_standard_reranking_instance_identity():
    """Test that evaluate_scorer does not detach the scorer from its init_tasks in standard reranking."""
    from unittest.mock import MagicMock
    from experimaestro.core.objects.config_utils import ConfigWrapper

    mock_tests = MagicMock()
    mock_fs_factory = MagicMock()
    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        use_plaid=False,
    )
    evaluate_scorer(
        cfg=cfg,
        cfg_tags={},
        tests=mock_tests,
        first_stage_factories={"splade": (mock_fs_factory, [])},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )
    assert mock_tests.evaluate_retriever.call_count == 1
    call_args = mock_tests.evaluate_retriever.call_args
    retriever_partial = call_args[0][0]
    init_tasks = call_args[1]["init_tasks"]

    # Crucial assertion: the scorer passed to scorer_retriever MUST BE IDENTICAL to the one in init_tasks
    evaluated_scorer = retriever_partial.keywords["scorer"]
    loader_task = init_tasks[0]
    raw_loader = (
        loader_task.value
        if isinstance(loader_task, ConfigWrapper)
        else getattr(loader_task, "task", loader_task)
    )
    assert evaluated_scorer is raw_loader.value


def test_evaluate_scorer_baseline_comparison_instance_identities():
    """Test that evaluate_scorer keeps proper instance identities for both baseline and PLAID evaluations."""
    from unittest.mock import MagicMock
    from experimaestro.core.objects.config_utils import ConfigWrapper

    mock_tests = MagicMock()
    mock_fs_factory = MagicMock()
    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        use_plaid=True,
        compare_with_baseline=True,
    )
    evaluate_scorer(
        cfg=cfg,
        cfg_tags={},
        tests=mock_tests,
        first_stage_factories={"splade": (mock_fs_factory, [])},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )
    # Should evaluate both baseline reranker and PLAID
    assert mock_tests.evaluate_retriever.call_count == 2

    # Call 0: Baseline 2-stage reranker
    call_0 = mock_tests.evaluate_retriever.call_args_list[0]
    retriever_factory_0 = call_0[0][0]
    scorer_0 = retriever_factory_0.keywords["scorer"]
    init_tasks_0 = call_0[1]["init_tasks"]
    raw_loader_0 = (
        init_tasks_0[0].value
        if isinstance(init_tasks_0[0], ConfigWrapper)
        else getattr(init_tasks_0[0], "task", init_tasks_0[0])
    )
    assert scorer_0 is raw_loader_0.value
    # Scorer does not have plaid_retriever, prune_threshold, or pref_attn_implementation tag
    assert "plaid_retriever" not in scorer_0.tags()
    assert "prune_threshold" not in scorer_0.tags()
    assert "pref_attn_implementation" not in scorer_0.tags()

    # Call 1: PLAID retriever
    call_1 = mock_tests.evaluate_retriever.call_args_list[1]
    plaid_factory_1 = (
        call_1.kwargs["retriever"] if "retriever" in call_1.kwargs else call_1[0][0]
    )
    scorer_1 = plaid_factory_1.mice_model
    init_tasks_1 = (
        call_1.kwargs["init_tasks"]
        if "init_tasks" in call_1.kwargs
        else call_1[1]["init_tasks"]
    )
    raw_loader_1 = (
        init_tasks_1[0].value
        if isinstance(init_tasks_1[0], ConfigWrapper)
        else getattr(init_tasks_1[0], "task", init_tasks_1[0])
    )
    assert scorer_1 is raw_loader_1.value
    assert "plaid_retriever" not in scorer_1.tags()

    # Crucial assertion: Scorer is NOT cloned! Both baseline and PLAID share the exact same instance!
    assert scorer_0 is scorer_1
    assert raw_loader_0.value is raw_loader_1.value


def test_two_stage_reranker_tags_plaid_retriever_false():
    """Test that two_stage_reranker helper tags the created retriever with plaid_retriever=False."""
    from unittest.mock import MagicMock, patch
    from Evaluation.experiment import two_stage_reranker

    mock_retriever = MagicMock()
    mock_retriever.tag.return_value = mock_retriever
    with patch("Evaluation.experiment.scorer_retriever", return_value=mock_retriever):
        res = two_stage_reranker(
            documents=MagicMock(),
            batch_size=16,
            scorer=MagicMock(),
            retrievers=MagicMock(),
            key="test_dataset",
        )
        mock_retriever.tag.assert_called_once_with("plaid_retriever", False)
        assert res is mock_retriever


def test_scorer_tags_only_include_grid_parameters():
    """Test that prune_threshold and attn_implementation are NOT tagged by default, but ARE tagged when grid-searched."""
    from unittest.mock import MagicMock

    mock_tests = MagicMock()
    mock_fs_factory = MagicMock()
    cfg = BaselinesConfig(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        scorer_type="miceV2",
        use_plaid=False,
    )

    # 1. Non-grid run: prune_threshold, attn_implementation, plaid_retriever must NOT be in tags
    evaluate_scorer(
        cfg=cfg,
        cfg_tags={},
        tests=mock_tests,
        first_stage_factories={"splade": (mock_fs_factory, [])},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )
    call_args = mock_tests.evaluate_retriever.call_args
    scorer_default = call_args[0][0].keywords["scorer"]
    tags_default = scorer_default.tags()
    assert "prune_threshold" not in tags_default
    assert "pref_attn_implementation" not in tags_default
    assert "plaid_retriever" not in tags_default
    assert tags_default["scorer"] == "VictorMorand/MiceV2-l10p4-ettin-68m"

    # 2. Grid-searched run: parameters present in cfg_tags MUST be tagged
    mock_tests.reset_mock()
    evaluate_scorer(
        cfg=cfg,
        cfg_tags={"hard_prune_threshold": 0.5, "pref_attn_implementation": "sdpa"},
        tests=mock_tests,
        first_stage_factories={"splade": (mock_fs_factory, [])},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )
    call_args = mock_tests.evaluate_retriever.call_args
    scorer_grid = call_args[0][0].keywords["scorer"]
    tags_grid = scorer_grid.tags()
    assert tags_grid["hard_prune_threshold"] == 0.5
    assert tags_grid["pref_attn_implementation"] == "sdpa"
    assert "plaid_retriever" not in tags_grid


def test_evaluate_scorer_local_st_directory_as_scorer_id(tmp_path):
    """Test that evaluate_scorer treats a local directory path with scorer_type='st' as a HF/ST model rather than an Experimaestro object."""
    import json
    from unittest.mock import MagicMock
    from xpmir.neural.sentence_transformers import STCrossEncoder

    # Create mock ST model directory on disk with NO experimaestro.json
    model_dir = tmp_path / "cross-encoder-ettin-68m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 512}))
    (model_dir / "modules.json").write_text(json.dumps([]))

    mock_tests = MagicMock()
    mock_fs_factory = MagicMock()
    cfg = BaselinesConfig(
        scorer_id=str(model_dir),
        scorer_type="st",
        max_query_length=32,
        max_doc_length=512,
    )

    evaluate_scorer(
        cfg=cfg,
        cfg_tags={},
        tests=mock_tests,
        first_stage_factories={"splade": (mock_fs_factory, [])},
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )

    assert mock_tests.evaluate_retriever.call_count == 1
    call_args = mock_tests.evaluate_retriever.call_args
    retriever_partial = call_args[0][0]
    evaluated_scorer = retriever_partial.keywords["scorer"]
    init_tasks = call_args[1]["init_tasks"]

    # Verify that it is instantiated as an STCrossEncoder with no init tasks and no from_experiment tags
    assert isinstance(evaluated_scorer, STCrossEncoder)
    assert init_tasks == []
    tags = evaluated_scorer.tags()
    assert tags["scorer"] == str(model_dir)
    assert tags["scorer_type"] == "stCrossEncoder"
    assert "from_experiment" not in tags
    assert "from_run_id" not in tags


def test_get_hf_config_local_directory(tmp_path):
    """Test that get_hf_config reads config.json from a local directory directly."""
    import json
    from xpm_torch.huggingface import get_hf_config

    model_dir = tmp_path / "local_hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 256, "hidden_size": 768}))

    config = get_hf_config(str(model_dir))
    assert config["max_position_embeddings"] == 256
    assert config["hidden_size"] == 768


def test_prepare_hf_model_local_directory(tmp_path):
    """Test that prepare_hf_model returns True immediately for a local directory."""
    from xpm_torch.huggingface import prepare_hf_model

    local_dir = tmp_path / "dummy_dir"
    local_dir.mkdir()
    assert prepare_hf_model(str(local_dir)) is True


def test_baselines_config_precompute_first_stage():
    """Verify that BaselinesConfig supports precompute_first_stage defaulting to True."""
    cfg_default = BaselinesConfig()
    assert cfg_default.precompute_first_stage is True

    cfg_no_precompute = BaselinesConfig(precompute_first_stage=False)
    assert cfg_no_precompute.precompute_first_stage is False


def test_evaluate_scorer_with_live_retriever_factory_when_not_precomputed():
    """Verify that evaluate_scorer works cleanly and passes retriever_init_tasks when first_stage_factories contains a live retriever factory."""
    from unittest.mock import MagicMock
    from xpmir.rankers.scorer import Scorer

    mock_tests = MagicMock()
    mock_retriever_factory = MagicMock()
    dummy_init_task = MagicMock()

    cfg = BaselinesConfig(
        scorer_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer_type="stCrossEncoder",
        use_plaid=False,
        precompute_first_stage=False,
    )
    first_stage_factories = {
        "splade": (mock_retriever_factory, [dummy_init_task])
    }

    evaluate_scorer(
        cfg=cfg,
        cfg_tags={},
        tests=mock_tests,
        first_stage_factories=first_stage_factories,
        launcher_evaluate=None,
        launcher_index=None,
        fabric_config=None,
    )

    assert mock_tests.evaluate_retriever.call_count == 1
    call_args = mock_tests.evaluate_retriever.call_args
    retriever_partial = call_args[0][0]
    # Check that retrievers passed to two_stage_reranker is indeed the live mock_retriever_factory
    assert retriever_partial.keywords["retrievers"] is mock_retriever_factory
    # Check that retriever_init_tasks was passed to evaluate_retriever
    assert dummy_init_task in call_args.kwargs["init_tasks"]
