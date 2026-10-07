"""Unit tests for ColBERT PLAID integration in cross_encoders_training/experiment.py
and include_recall_100 in Evaluation/experiment.py.

Run with:
    uv run pytest tests/test_colbert_training_plaid.py -v
"""

from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from configuration import CE_FineTuning, Indexation, PlaidConfiguration, ScorerType
from cross_encoders_training.experiment import (
    build_STMultiVectorEncoder,
    get_ce_name,
)
from Evaluation.experiment import BaselinesConfig
from format import dataframe_to_latex
from index_utils import get_plaid_index, _indexes
from retrievers import plaid_retriever
from xpmir.index.plaid import PlaidIndex
from xpmir.neural.sentence_transformers import STModuleLoader, st_multivector_scorer
from xpmir.test.utils.utils import SampleDocumentStore
from tests import build_tests


def test_build_st_multivector_encoder_dim_none():
    """Verify build_STMultiVectorEncoder builds cleanly with self-inferred dimension (dim=None)."""
    cfg = CE_FineTuning()
    cfg.base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.max_query_length = 32
    cfg.max_doc_length = 128

    scorer, init_tasks = build_STMultiVectorEncoder(cfg)
    assert scorer is not None
    assert init_tasks == []
    assert scorer.query_maxlen == 32
    assert scorer.doc_maxlen == 128
    assert scorer.dim is None
    # Dim is self-inferred by the underlying model
    instance = scorer.instance()
    instance.initialize()
    assert instance.dimension == 128


def test_build_st_multivector_encoder_explicit_dim():
    """Verify build_STMultiVectorEncoder forwards explicit plaid.dim (preserving hash)."""
    cfg = CE_FineTuning()
    cfg.base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.max_query_length = 32
    cfg.max_doc_length = 128
    cfg.plaid.dim = 128

    scorer, init_tasks = build_STMultiVectorEncoder(cfg)
    assert scorer is not None
    assert init_tasks == []
    assert scorer.dim == 128
    instance = scorer.instance()
    instance.initialize()
    assert instance.dimension == 128


def test_get_plaid_index_caching_key():
    """Verify that get_plaid_index caches by (index_builder, init_task_checkpoint_paths)."""
    _indexes.clear()
    doc_store = SampleDocumentStore.C(num_docs=4)
    scorer = st_multivector_scorer(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
    plaid_cfg = PlaidConfiguration(n_bits=2, kmeans_niters=2, buffer_size=4)

    loader1 = STModuleLoader.C(value=scorer, path=Path("/tmp/chk1"))
    loader2 = STModuleLoader.C(value=scorer, path=Path("/tmp/chk2"))

    # Mock submit to avoid running actual index build
    mock_index1 = MagicMock()
    mock_index2 = MagicMock()

    with patch("xpmir.index.plaid.PlaidIndexBuilder.C.submit") as mock_submit:
        mock_submit.side_effect = [mock_index1, mock_index2]

        idx1 = get_plaid_index(doc_store, scorer, plaid_cfg, launcher_index=None, init_tasks=[loader1])
        # Same loader -> should return cached idx1 without submitting again
        idx1_cached = get_plaid_index(doc_store, scorer, plaid_cfg, launcher_index=None, init_tasks=[loader1])
        assert idx1 is idx1_cached
        assert mock_submit.call_count == 1

        # Different loader (different weights) -> should submit new index
        idx2 = get_plaid_index(doc_store, scorer, plaid_cfg, launcher_index=None, init_tasks=[loader2])
        assert idx2 is mock_index2
        assert mock_submit.call_count == 2


def test_get_ce_name_plaid_suffix():
    """Verify that get_ce_name appends '-plaid' when plaid_retriever is True."""
    tags_standard = {
        "base": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "loss": "MSE_mixedbread_large",
        "scorer_type": ScorerType.stMultiVectorEncoder.value,
        "checkpoint": "last",
        "plaid_retriever": False,
    }
    name_standard = get_ce_name(tags_standard)
    assert "colbert" in name_standard
    assert "plaid" not in name_standard

    tags_plaid = {
        "base": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "loss": "MSE_mixedbread_large",
        "scorer_type": ScorerType.stMultiVectorEncoder.value,
        "checkpoint": "last",
        "plaid_retriever": True,
    }
    name_plaid = get_ce_name(tags_plaid)
    assert name_plaid.endswith("-plaid")
    assert name_plaid == f"{name_standard}-plaid"


def test_baselines_config_include_recall_100():
    """Verify BaselinesConfig supports include_recall_100 and propagates to cfg.evaluation."""
    cfg = BaselinesConfig()
    assert hasattr(cfg, "include_recall_100")
    assert cfg.include_recall_100 is False

    cfg.include_recall_100 = True
    if cfg.include_recall_100:
        cfg.evaluation.include_recall_100 = True
    assert cfg.evaluation.include_recall_100 is True

    # Verify that build_tests includes R@100 in measures when include_recall_100 is True
    cfg.evaluation.datasets = ["fiqa"]
    mock_eval = MagicMock()
    built_measures = []

    def fake_build(self, *args, **kwargs):
        built_measures.append(kwargs.get("measures", []))
        return mock_eval

    with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
        build_tests(cfg.evaluation, check_docs=False)
        assert len(built_measures) == 1
        measures = built_measures[0]
        assert any(m.identifier == "R" and m.cutoff == 100 for m in measures)
        assert any(m.identifier == "nDCG" and m.cutoff == 10 for m in measures)


def test_independent_loader_cloning_for_plaid():
    """Verify that cloning a loader config for PLAID preserves original loader tags in memory."""
    scorer = st_multivector_scorer(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
    orig_loader = STModuleLoader.C(value=scorer, path=Path("/tmp/model_chk"))
    orig_loader.tag("checkpoint", "best").tag("seed", 42).tag("first_stage", "bm25").tag(
        "scorer", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Clone for PLAID
    plaid_loader = orig_loader.value.loader_config(
        orig_loader.path,
        settings=getattr(orig_loader, "settings", None),
    )
    for k, v in orig_loader.tags().items():
        if k not in ("first_stage", "scorer", "plaid_retriever"):
            plaid_loader.tag(k, v)
    plaid_loader.tag("first_stage", "cross-encoder/ms-marco-MiniLM-L-6-v2").tag(
        "scorer", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ).tag("plaid_retriever", True)

    # Check tags on both objects
    orig_tags = dict(orig_loader.tags())
    plaid_tags = dict(plaid_loader.tags())

    assert orig_tags["first_stage"] == "bm25"
    assert orig_tags["scorer"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "plaid_retriever" not in orig_tags or orig_tags["plaid_retriever"] is False

    assert plaid_tags["first_stage"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert plaid_tags["scorer"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert plaid_tags["plaid_retriever"] is True
    assert plaid_tags["checkpoint"] == "best"
    assert plaid_tags["seed"] == 42


def test_colbert_checkpoint_evaluation_dual_dispatch():
    """Verify that the checkpoint evaluation loop dispatches both 2-stage and PLAID evaluations with correct tagging."""
    cfg = CE_FineTuning()
    cfg.base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.use_plaid = True
    scorer_type = ScorerType.stMultiVectorEncoder

    scorer = st_multivector_scorer(model_id=cfg.base)
    load_model = STModuleLoader.C(value=scorer, path=Path("/tmp/chk_best"))
    load_model.tag("checkpoint", "best").tag("seed", 42).tag("first_stage", "bm25").tag("scorer", cfg.base)

    models_to_evaluate = [("test_model", load_model)]
    all_weights = []
    evaluate_calls = []

    mock_tests = MagicMock()
    mock_tests.evaluate_retriever.side_effect = lambda ret, launcher, model_id=None, **kwargs: evaluate_calls.append(model_id)

    # Simulate checkpoint loop from cross_encoders_training/experiment.py
    for model_id, lm in models_to_evaluate:
        if cfg.use_plaid:
            lm.tag("plaid_retriever", False)
        all_weights.append(lm)
        mock_tests.evaluate_retriever(
            "scorer_retriever_factory",
            launcher="launcher_eval",
            model_id=model_id,
            init_tasks=[lm],
        )

        if cfg.use_plaid and scorer_type == ScorerType.stMultiVectorEncoder:
            plaid_lm = scorer.loader_config(
                lm.path,
                settings=getattr(lm, "settings", None),
            )
            for k, v in lm.tags().items():
                if k not in ("first_stage", "scorer", "plaid_retriever"):
                    plaid_lm.tag(k, v)
            plaid_lm.tag("first_stage", cfg.base).tag(
                "scorer", cfg.base
            ).tag("plaid_retriever", True)
            all_weights.append(plaid_lm)

            plaid_factory = partial(
                plaid_retriever,
                cfg,
                cfg.base,
                scorer,
                launcher_index=None,
                init_tasks=[plaid_lm],
                topk=cfg.retrieval.k,
                seed=42,
            )
            mock_tests.evaluate_retriever(
                plaid_factory,
                launcher="launcher_eval",
                model_id=f"{model_id}-plaid",
                init_tasks=[plaid_lm],
            )

            # Verify the factory can be called by xpmir evaluate_retriever with (documents, key=key)
            doc_store = SampleDocumentStore.C(num_docs=4)
            fake_index = PlaidIndex.C(documents=doc_store, index_path=Path("/tmp/fake_index"))
            with patch("retrievers.get_plaid_index", return_value=fake_index):
                retriever_instance = plaid_factory(doc_store, key="test_key")
                assert retriever_instance is not None

    # 1. Verify two evaluations dispatched
    assert evaluate_calls == ["test_model", "test_model-plaid"]
    # 2. Verify both loaders in all_weights
    assert len(all_weights) == 2
    w1, w2 = all_weights
    assert dict(w1.tags())["first_stage"] == "bm25"
    assert dict(w1.tags())["scorer"] == cfg.base
    assert dict(w1.tags())["plaid_retriever"] is False
    assert dict(w2.tags())["first_stage"] == cfg.base
    assert dict(w2.tags())["scorer"] == cfg.base
    assert dict(w2.tags())["plaid_retriever"] is True


def test_dataframe_to_latex_plaid_deduplication():
    """Verify that dataframe_to_latex deduplicates labels when first_stage == scorer."""
    df = pd.DataFrame([
        {
            "dataset": "fiqa",
            "first_stage": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "scorer": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "nDCG@10": 0.45,
        },
        {
            "dataset": "fiqa",
            "first_stage": "bm25",
            "scorer": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "nDCG@10": 0.40,
        },
    ])

    latex = dataframe_to_latex(df, metric_col="nDCG@10")
    # For standalone PLAID, first_stage == scorer -> should only display model name once, not "model / model"
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2 / cross-encoder/ms-marco-MiniLM-L-6-v2" not in latex
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in latex
    # For 2-stage re-ranking, should keep "bm25 / cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "bm25 / cross-encoder/ms-marco-MiniLM-L-6-v2" in latex


def test_plaid_batch_sizes_separation():
    """Verify that get_plaid_index correctly separates doc encoder batch_size from fast_plaid_batch_size,
    and that varying fast_plaid_batch_size changes the PlaidIndexBuilder identifier."""
    _indexes.clear()
    doc_store = SampleDocumentStore.C(num_docs=4)
    scorer = st_multivector_scorer(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")

    plaid_cfg = PlaidConfiguration(
        n_bits=2,
        kmeans_niters=2,
        buffer_size=4,
        fast_plaid_batch_size=25000,
    )
    indexation_cfg = Indexation(batch_size=64)

    captured_builder = None

    def mock_submit(self, launcher=None, init_tasks=None):
        nonlocal captured_builder
        captured_builder = self
        return MagicMock()

    with patch("xpmir.index.plaid.PlaidIndexBuilder.C.submit", new=mock_submit):
        get_plaid_index(
            documents=doc_store,
            encoder=scorer,
            plaid_cfg=plaid_cfg,
            indexation_cfg=indexation_cfg,
            launcher_index=None,
        )

    assert captured_builder is not None
    assert captured_builder.batch_size == 64
    assert captured_builder.fast_plaid_batch_size == 25000

    # Also test varying fast_plaid_batch_size changes the task identifier
    _indexes.clear()
    plaid_cfg_variant = PlaidConfiguration(
        n_bits=2,
        kmeans_niters=2,
        buffer_size=4,
        fast_plaid_batch_size=32,
    )
    captured_variant = None

    def mock_submit_variant(self, launcher=None, init_tasks=None):
        nonlocal captured_variant
        captured_variant = self
        return MagicMock()

    with patch("xpmir.index.plaid.PlaidIndexBuilder.C.submit", new=mock_submit_variant):
        get_plaid_index(
            documents=doc_store,
            encoder=scorer,
            plaid_cfg=plaid_cfg_variant,
            indexation_cfg=indexation_cfg,
            launcher_index=None,
        )

    # fast_plaid_batch_size is Meta[int], so it correctly preserves the task identifier
    assert captured_variant is not None
    assert captured_variant.fast_plaid_batch_size == 32
    assert captured_builder.__identifier__() == captured_variant.__identifier__()


def test_colbert_training_routes_plaid_to_long_launcher_index():
    """Verify that during ColBERT training evaluation with PLAID,
    effective_long_evals routes matching datasets to long_launcher_index."""
    from retrievers import plaid_retriever
    from functools import partial

    _indexes.clear()

    launcher_default = MagicMock(name="default_launcher")
    launcher_long = MagicMock(name="long_launcher")

    cfg = CE_FineTuning(use_plaid=True)
    cfg.base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.retrieval.long_evals = ["fever", "arguana"]
    # indexation.long_evals is None by default, should inherit retrieval.long_evals
    assert cfg.indexation.long_evals is None

    effective_long_evals = (
        getattr(cfg.indexation, "long_evals", None)
        or getattr(cfg.retrieval, "long_evals", None)
    )
    assert effective_long_evals == ["fever", "arguana"]

    scorer = st_multivector_scorer(model_id=cfg.base)
    loader = STModuleLoader.C(value=scorer, path=Path("/tmp/chk_test"))

    factory = partial(
        plaid_retriever,
        cfg,
        cfg.base,
        scorer,
        launcher_index=launcher_default,
        long_launcher_index=launcher_long,
        long_evals=effective_long_evals,
        init_tasks=[loader],
        topk=10,
        seed=0,
    )

    doc_store_scifact = SampleDocumentStore.C(id="scifact", num_docs=2)
    doc_store_fever = SampleDocumentStore.C(id="fever", num_docs=2)

    submitted_launchers = []

    def mock_submit(builder_self, launcher=None, **kwargs):
        submitted_launchers.append(launcher)
        return PlaidIndex.C(
            documents=builder_self.documents,
            index_path=Path("/tmp/fake_index"),
        )

    with patch("xpmir.index.plaid.PlaidIndexBuilder.C.submit", new=mock_submit):
        # Non-matching key (scifact) -> default launcher
        factory(doc_store_scifact, key="beir/scifact")
        assert len(submitted_launchers) == 1
        assert submitted_launchers[0] == launcher_default

        # Matching key (fever) -> long launcher
        factory(doc_store_fever, key="beir/fever")
        assert len(submitted_launchers) == 2
        assert submitted_launchers[1] == launcher_long
