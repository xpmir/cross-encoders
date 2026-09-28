"""
Unit tests for experiment model resolver, decoupled config validation, and parameter override warnings.
Run with: `uv run pytest tests/test_experiment_model_resolver.py -v`
"""

import logging
from pathlib import Path
import pytest

from training_utils import (
    find_experiment_results_model_dir,
    resolve_trained_model_loader,
    validate_and_align_config,
)
from configuration import CE_FineTuning
from MICE.experiments.micev2_training import MiceV2_FineTuning, build_mice_v2


def test_find_experiment_results_model_dir_direct_path(tmp_path):
    """Test resolving a direct path to a model folder containing experimaestro.json."""
    model_dir = tmp_path / "results" / "models" / "TestModel"
    model_dir.mkdir(parents=True)
    (model_dir / "experimaestro.json").write_text("{}")

    # Pass the direct model directory
    resolved = find_experiment_results_model_dir(str(model_dir))
    assert resolved == model_dir

    # Pass the parent run directory
    resolved_run = find_experiment_results_model_dir(str(tmp_path))
    assert resolved_run == model_dir


def test_find_experiment_results_model_dir_not_found():
    """Test that a non-existent experiment raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found in any workspace"):
        find_experiment_results_model_dir("non_existent_experiment_xyz_12345")


def test_resolve_trained_model_loader_fallback_to_hub():
    """Test that a Hugging Face repo ID candidate that is not a local experiment falls back to Hub."""
    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    assert loader is not None
    assert hasattr(loader, "model")
    assert loader.model.hf_id == "cross-encoder/ettin-reranker-68m-v1"


def test_resolve_local_experiment_if_present():
    """Test resolving local experiment 'train_mice_ettin32m_l4+4' if present on the machine."""
    try:
        resolved = find_experiment_results_model_dir("train_mice_ettin32m_l4+4")
        assert resolved.exists()
        assert (resolved / "experimaestro.json").exists()
        assert "Mice-l4+4-ettin-32m" in str(resolved)

        loader = resolve_trained_model_loader("train_mice_ettin32m_l4+4")
        assert loader is not None
        assert hasattr(loader, "model")
        assert loader.model.hf_id == "jhu-clsp/ettin-encoder-32m"
        assert loader.model.n_contextualization_layers == 4
        assert loader.model.n_interaction_layers == 4
    except FileNotFoundError:
        pytest.skip("train_mice_ettin32m_l4+4 experiment not available on this machine")


def test_ce_finetuning_from_experiment_field():
    """Test CE_FineTuning configuration supports from_experiment."""
    cfg = CE_FineTuning(from_experiment="train_mice_ettin32m_l4+4")
    assert cfg.from_experiment == "train_mice_ettin32m_l4+4"


def test_init_micev2_from_experiment_checkpoint():
    """Test InitMICEV2FromCheckpoint execution using an exported model loader from an experiment/hub."""
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        utility_pred=True,
        hard_pruning=True,
        hard_prune_threshold=0.5,
    )
    for task in init_tasks:
        task.instance().execute()

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    for param in scorer.hf_model.utility_predictor.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.top_layers.parameters():
        assert param.requires_grad


def test_validate_and_align_inherits_default_parameters(caplog):
    """Test that default parameters inherit from the loaded model with INFO logs."""
    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    tags = {}
    loader = resolve_trained_model_loader(cfg.from_experiment)

    with caplog.at_level(logging.INFO):
        validate_and_align_config(cfg, loader.model, tags=tags, from_ref=cfg.from_experiment)

    # Verified inheritance: 10 contextualization and 4 interaction layers inherited from 68m model
    assert cfg.base == "cross-encoder/ettin-reranker-68m-v1"
    assert cfg.n_contextualization_layers == 10
    assert cfg.n_interaction_layers == 4
    assert tags["base"] == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["from_experiment"] == "VictorMorand/MiceV2-l10p4-ettin-68m"

    assert "[Model Alignment] Inherited base" in caplog.text
    assert "[Model Alignment] Inherited 'n_contextualization_layers'" in caplog.text
    assert "[PARAM OVERRIDE]" not in caplog.text


def test_validate_and_align_warns_on_explicit_override(caplog):
    """Test that explicitly set parameters differing from loaded model emit loud WARNINGs."""
    cfg = MiceV2_FineTuning(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        base="jhu-clsp/ettin-encoder-17m",
        n_interaction_layers=2,  # Loaded model has 4
    )
    tags = {}
    loader = resolve_trained_model_loader(cfg.from_experiment)

    with caplog.at_level(logging.INFO):
        validate_and_align_config(cfg, loader.model, tags=tags, from_ref=cfg.from_experiment)

    # User-specified values should be preserved
    assert cfg.base == "jhu-clsp/ettin-encoder-17m"
    assert cfg.n_interaction_layers == 2

    # Loud warnings must be logged
    assert "[PARAM OVERRIDE] Config explicitly overrides base model!" in caplog.text
    assert (
        "[PARAM OVERRIDE] Config explicitly overrides loaded model parameter 'n_interaction_layers'!"
        in caplog.text
    )


def test_build_mice_v2_with_from_experiment():
    """Test that build_mice_v2 automatically loads checkpoint and aligns config when from_experiment is set."""
    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    tags = {}

    scorer, init_tasks = build_mice_v2(cfg, cfg_tags=tags)

    # Checks base and tags alignment
    assert cfg.base == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["base"] == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["from_experiment"] == "VictorMorand/MiceV2-l10p4-ettin-68m"

    # Checks init task is InitMICEV2FromCheckpoint
    assert len(init_tasks) == 1
    assert "InitMICEV2FromCheckpoint" in init_tasks[0].__class__.__name__


def test_modernbert_mice_scorer_warns_on_override_from_checkpoint(caplog):
    """Test that ModernBertMiceScorer.__initialize__ logs [MICE Config Override] warnings."""
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=4,
        top_layers_doc_ln=True,
        bottom_final_norm=False,
        utility_temp=2.0,  # Overriding checkpoint's 1.0
        hard_prune_threshold=0.3,  # Overriding checkpoint's 0.5
    )

    with caplog.at_level(logging.WARNING):
        for task in init_tasks:
            task.instance().execute()

    assert "[MICE Config Override] Overriding checkpoint parameter" in caplog.text
    assert "utility_temp" in caplog.text
    assert "hard_prune_threshold" in caplog.text


def test_modernbert_mice_scorer_structural_mismatch_raises(caplog):
    """Test that overriding structural compress_dim logs warnings and fails weight loading."""
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=2,  # Overriding checkpoint's 4 layers
        compress_dim=2.0,  # Overriding checkpoint's 1.0 compression
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            for task in init_tasks:
                task.instance().execute()

    assert "[MICE Config Override] Overriding checkpoint parameter" in caplog.text
    assert "compress_dim" in caplog.text


def test_checkpoint_loader_taskstub_job_serialization():
    """Test that an init task with a loaded checkpoint_loader (containing a TaskStub)
    can be serialized for job execution via state_dict/outputjson without error."""
    from experimaestro import state_dict
    from experimaestro.core.context import SerializationContext
    from experimaestro.core.objects.config import TaskStub
    from MICE.experiments.micev2_training import MiceV2_FineTuning, build_mice_v2

    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    scorer, init_tasks = build_mice_v2(cfg, cfg_tags={})

    assert len(init_tasks) == 1
    init_task = init_tasks[0]
    checkpoint_loader = init_task.checkpoint_loader

    # Verify checkpoint_loader.task is a TaskStub
    assert isinstance(checkpoint_loader.__xpm__.task, TaskStub)

    # Test serialization of init_task (as happens when scheduler writes params.json)
    context = SerializationContext(save_directory=None)
    init_task.__xpm__.seal(context)
    data = state_dict(context, [init_task])
    assert data is not None

    # Verify TaskStub was serialized with stub=True
    stub_objects = [obj for obj in data["objects"] if obj.get("stub")]
    assert len(stub_objects) == 1
    assert stub_objects[0]["id"] == id(checkpoint_loader.__xpm__.task)


def test_init_micev2_from_checkpoint_with_compression():
    """Test that InitMICEV2FromCheckpoint discards top layer seeding when compress_dim > 1.0,
    loading bottom layers from the uncompressed checkpoint and keeping compressed top layers
    at fresh random initialization without PyTorch shape mismatch errors."""
    import torch
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    # 1. Resolve uncompressed checkpoint (VictorMorand/MiceV2-l10p4-ettin-68m has 512-dim layers)
    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    # 2. Build target model with compress_dim = 1.333333333 (top layers hidden_size = 384)
    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.333333333,
        utility_pred=True,
    )

    # 3. Create InitMICEV2FromCheckpoint task
    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    # 4. Execute: must succeed without size mismatch exception
    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    hf_model = scorer.hf_model

    # Bottom layers: 512-dim, frozen
    assert hf_model.model.layers[0].attn.Wqkv.weight.shape == torch.Size([1536, 512])
    for param in hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in hf_model.model.embeddings.parameters():
        assert not param.requires_grad

    # Top interaction layers: 384-dim, trainable
    assert hf_model.top_layers[0].self_attn.Wqkv.weight.shape == torch.Size([1152, 384])
    assert hf_model.adapter.weight.shape == torch.Size([384, 512])
    assert hf_model.output_modules.dense_score.linear.weight.shape == torch.Size([1, 384])
    for param in hf_model.top_layers.parameters():
        assert param.requires_grad
    assert hf_model.adapter.weight.requires_grad
    assert hf_model.output_modules.dense_score.linear.weight.requires_grad


def test_init_micev2_from_checkpoint_with_compress_init_random_uncompressed():
    """Test that compress_dim=1.0 with compress_init_option='random' re-initializes top layers
    and head with fresh random weights while preserving checkpoint bottom layers."""
    import torch
    from safetensors.torch import load_file
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.0,
        compress_init_option="random",
        utility_pred=True,
    )

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    hf_model = scorer.hf_model

    sd = load_file(loader.path)

    # 1. Bottom layers: MUST match checkpoint and be frozen
    assert torch.equal(hf_model.model.layers[0].attn.Wqkv.weight, sd["model.layers.0.attn.Wqkv.weight"])
    for param in hf_model.model.layers.parameters():
        assert not param.requires_grad

    # 2. Top interaction layers and classification head: MUST NOT match checkpoint (fresh random weights) and be trainable
    assert not torch.equal(hf_model.top_layers[0].self_attn.Wqkv.weight, sd["top_layers.0.self_attn.Wqkv.weight"])
    for param in hf_model.top_layers.parameters():
        assert param.requires_grad
    assert hf_model.output_modules.dense_score.linear.weight.requires_grad

    # 3. Tokenizer: MUST be initialized in main process and functional
    assert scorer.tokenizer._initialized
    assert hasattr(scorer.tokenizer, "tokenizer") and scorer.tokenizer.tokenizer is not None

    from xpmir.letor.records import PairwiseItems, PairwiseItem
    from types import SimpleNamespace
    p_item = PairwiseItem(
        query={"id": "q1", "text_item": SimpleNamespace(text="test query")},
        positive={"id": "d1", "text_item": SimpleNamespace(text="test doc 1")},
        negative={"id": "d2", "text_item": SimpleNamespace(text="test doc 2")},
    )
    tokenized = scorer.batch_tokenize(PairwiseItems([p_item]))
    assert tokenized.tokenized_q.ids.shape[0] == 2
    assert tokenized.tokenized_docs.ids.shape[0] == 2


def test_init_micev2_from_checkpoint_with_compress_init_seed_uncompressed():
    """Test that compress_dim=1.0 with compress_init_option='seed' preserves all checkpoint weights (top and bottom)."""
    import torch
    from safetensors.torch import load_file
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.0,
        compress_init_option="seed",
        utility_pred=True,
    )

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    hf_model = scorer.hf_model

    sd = load_file(loader.path)

    # Bottom layers, top layers, and classifier must all match checkpoint
    assert torch.equal(hf_model.model.layers[0].attn.Wqkv.weight, sd["model.layers.0.attn.Wqkv.weight"])
    assert torch.equal(hf_model.top_layers[0].self_attn.Wqkv.weight, sd["top_layers.0.self_attn.Wqkv.weight"])
    assert torch.equal(hf_model.output_modules.dense_score.linear.weight, sd["output_modules.dense_score.linear.weight"])


def test_load_model_raises_on_shape_mismatch():
    """Test that ModernBertMiceScorer.load_model strictly raises RuntimeError on weight shape mismatch."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, _ = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=4,
        compress_dim=1.0,
    )
    scorer = scorer_cfg.instance()
    scorer.initialize()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create incompatible weights: layer 0 attention weight has wrong shape [100, 100] instead of [1536, 512]
        bad_sd = {"model.layers.0.attn.Wqkv.weight": torch.randn(100, 100)}
        save_file(bad_sd, str(tmp_path / "model.safetensors"))

        with pytest.raises(RuntimeError) as excinfo:
            scorer.load_model(tmp_path)
        assert "size mismatch" in str(excinfo.value)


def test_load_model_ignore_top():
    """Test that ModernBertMiceScorer.load_model with ignore_top=True discards top weights and succeeds."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model
    uncompressed_model_dir = Path(loader.path).parent

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.333333333,
    )
    scorer = scorer_cfg.instance()
    scorer.initialize()

    # With ignore_top=False (default), uncompressed top layers mismatch shape [512] vs [384] -> raises RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        scorer.load_model(uncompressed_model_dir, ignore_top=False)
    assert "size mismatch" in str(excinfo.value)

    # With ignore_top=True, top layers are discarded -> succeeds cleanly
    scorer.load_model(uncompressed_model_dir, ignore_top=True)
    assert scorer._initialized
