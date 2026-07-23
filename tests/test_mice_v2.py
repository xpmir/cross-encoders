"""
Tests for ModernBERT MICE V2 Architecture.
Run with: `uv run pytest tests/test_mice_v2.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch
from sentence_transformers import CrossEncoder

from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, ModernBertMiceScorer
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param


class TestMiceV2ForwardTask(LightweightTask):
    """Tests Mice V2 Loading, scoring a PointWiseItem, and reload persistence"""

    scorer: Param[ModernBertMiceScorer]

    def execute(self):
        self.scorer.initialize()

        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(input_records)

        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
@pytest.mark.parametrize(
    "n_contextualization_layers, n_docs_ctx_layers, bound_bottom_layers",
    [
        (2, None, True),
        (2, 3, False),
    ],
)
def test_mice_v2(
    model_id,
    n_contextualization_layers,
    n_docs_ctx_layers,
    bound_bottom_layers,
):
    """Tests MICE V2 forward pass, weight persistence, and SentenceTransformers export."""

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        bound_bottom_layers=bound_bottom_layers,
    )

    test_forward_task = TestMiceV2ForwardTask.C(scorer=scorer_cfg).instance()

    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()

    scorer = test_forward_task.scorer

    # --- Persistence Verification ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"
        st_save_path = tmp_path / "st_model"

        # 1. Save and reload scorer model
        scorer.save_model(save_path)

        reloaded_scorer = ModernBertMiceScorer.C(
            hf_id=model_id,
            tokenizer=scorer_cfg.tokenizer,
            n_contextualization_layers=n_contextualization_layers,
        ).instance()
        reloaded_scorer.load_model(save_path)

        assert torch.allclose(
            scorer.hf_model.classifier.weight, reloaded_scorer.hf_model.classifier.weight
        ), "Classifier weights mismatch after standard reload!"

        # 2. SentenceTransformers Export & Import Roundtrip
        scorer.export_to_st(str(st_save_path))

        st_cross_encoder = CrossEncoder(str(st_save_path))
        scores = st_cross_encoder.predict([["What is the capital of France?", "Paris is the capital of France."]])
        assert scores is not None
        assert len(scores) == 1
