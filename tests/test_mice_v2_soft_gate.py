"""
Tests for MICE V2 Soft Gate Strategy & Utility Predictor.
Run with: `uv run pytest tests/test_mice_v2_soft_gate.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch

from MICE.modeling.mice_v2.modernbert_hf import (
    UtilityPredictor,
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    ModernBertMiceScorer,
    UtilityRegularizer,
)
from xpmir.letor.records import PointwiseItems
from xpm_torch.trainers import TrainerContext
from xpm_torch.metrics import Metrics
import logging

logging.basicConfig(level=logging.INFO)


def test_utility_predictor_direct():
    """Test UtilityPredictor module outputs, shapes, and value bounds."""
    batch_size, seq_len, d_model = 4, 32, 128
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    predictor = UtilityPredictor(hidden_dim=d_model)
    utilities = predictor(hidden_states)

    assert utilities.shape == (batch_size, seq_len, 1), f"Expected shape {(batch_size, seq_len, 1)}, got {utilities.shape}"
    assert (utilities >= 0.0).all() and (utilities <= 1.0).all(), "Utility scores must be strictly bounded in (0, 1)"

    # Test unpadded 1D input tensor [1, total_tokens, d_model]
    unpadded_hidden = torch.randn(1, 100, d_model)
    unpadded_utilities = predictor(unpadded_hidden)
    assert unpadded_utilities.shape == (1, 100, 1)


def test_utility_regularizer_unpadded_fa2():
    """Test UtilityRegularizer handling 1D unpadded FA2 utilities with 2D doc_mask."""
    reg_hook = UtilityRegularizer.C(lambda_reg=0.1)

    # 1D unpadded FA2 utilities [1, total_unpadded_tokens, 1] vs 2D padded doc_mask [batch_size, max_seqlen]
    fa2_utilities = torch.rand((1, 37720, 1))
    doc_mask = torch.ones((256, 1024), dtype=torch.long)

    class MockPath:
        pass

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )

    with context.losses() as losses:
        reg_hook(context, fa2_utilities, doc_mask=doc_mask)

    assert len(losses) == 1
    assert losses[0].name == "utility_reg"
    assert losses[0].value.item() == pytest.approx(fa2_utilities.mean().item())


def test_utility_regularizer_histogram_logging():
    """Test UtilityRegularizer calling writer.add_histogram when step matches histogram_interval."""
    reg_hook = UtilityRegularizer.C(lambda_reg=0.1, histogram_interval=100)

    utilities = torch.rand((4, 16, 1))
    doc_mask = torch.ones((4, 16), dtype=torch.long)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    histogram_calls = []

    class MockWriter:
        def add_histogram(self, tag, values, global_step=None):
            histogram_calls.append((tag, values.shape, global_step))

    context._writer = MockWriter()

    # Step 50: step % 100 != 0 -> should NOT log histogram
    context.state.steps = 50
    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask)
    assert len(histogram_calls) == 0

    # Step 100: step % 100 == 0 -> SHOULD log histogram
    context.state.steps = 100
    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask)
    assert len(histogram_calls) == 1
    assert histogram_calls[0][0] == "utility_distribution"
    assert histogram_calls[0][2] == 100


def test_modernbert_hf_soft_gate():
    """Test ModernBertMiceForSequenceClassification with utility_pred enabled."""
    config = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    query_ids = torch.randint(0, 1000, (2, 8))
    query_mask = torch.ones((2, 8), dtype=torch.long)
    doc_ids = torch.randint(0, 1000, (2, 16))
    doc_mask = torch.ones((2, 16), dtype=torch.long)

    with torch.no_grad():
        out = model(
            query_ids=query_ids,
            query_mask=query_mask,
            doc_ids=doc_ids,
            doc_mask=doc_mask,
        )

    assert model.last_utilities is not None, "model.last_utilities should be populated when utility_pred=True"
    assert model.last_utilities.shape == (2, 16, 1), f"Expected utilities shape (2, 16, 1), got {model.last_utilities.shape}"
    assert (model.last_utilities >= 0.0).all() and (model.last_utilities <= 1.0).all()


def test_modernbert_xpm_soft_gate_and_hook():
    """Test ModernBertMiceScorer with utility_pred=True and UtilityRegularizer hook."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    scorer.eval()

    queries = ["What is IR?", "How does soft gate work?"]
    documents = [
        "Information Retrieval is the activity of obtaining information resources relevant to an information need.",
        "Soft gating multiplies document token hidden states with predicted token utility scores.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        context = TrainerContext(
            logpath=tmp_path / "logs",
            path=tmp_path / "ckpts",
            max_epoch=1,
            steps_per_epoch=10,
            trainer=None,
            model=scorer,
            optimizer=None,
            fabric=None,
        )
        context.metrics = Metrics()
        reg_hook = UtilityRegularizer.C(lambda_reg=0.1)
        context.add_hook(reg_hook)

        with context.losses() as losses:
            output = scorer(input_records, info=context)
            assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"

        # Verify UtilityRegularizer registered loss in context
        assert len(losses) == 1, f"Expected 1 loss registered in context, got {len(losses)}"
        reg_loss_item = losses[0]
        assert reg_loss_item.name == "utility_reg"
        assert reg_loss_item.weight == 0.1
        assert reg_loss_item.value > 0.0

        # Verify metrics logged to context
        assert "utility_reg" in context.metrics.metrics
        assert "utility_mean" in context.metrics.metrics
        assert "utility_sparsity" in context.metrics.metrics


def test_soft_gate_save_load():
    """Test saving and loading a soft-gated MICE v2 model persistence."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir)
        scorer.save_model(save_path)

        # Verify config.json contains utility_pred=True
        reloaded_config = ModernBertMiceConfig.from_pretrained(save_path)
        assert getattr(reloaded_config, "utility_pred", False) is True, "utility_pred flag must persist in config.json"

        # Reload model weights into new instance
        reloaded_model = ModernBertMiceForSequenceClassification.from_pretrained(save_path)
        assert reloaded_model.utility_predictor is not None, "Reloaded model must instantiate utility_predictor"
