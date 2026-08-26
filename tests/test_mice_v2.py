"""
Tests for ModernBERT MICE V2 Architecture.
Run with: `uv run pytest tests/test_mice_v2.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, ModernBertMiceScorer
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

import logging
logging.basicConfig(level=logging.INFO)

class TestMiceV2ForwardTask(LightweightTask):
    """Tests Mice V2 Loading, scoring a PointWiseItem, and reload persistence"""

    scorer: Param[ModernBertMiceScorer]

    def execute(self):
        self.scorer.initialize()
        logging.info(f"Initialized MICE V2 scorer: {self.scorer}")
        queries = ["What is the capital of France?", "Where is Paris located?"]
        documents = [
            "Paris is the capital and most populous city of France.",
            "Paris is located in northern central France on the Seine river.",
        ]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(input_records)

        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"


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
    """Tests MICE V2 forward pass, weight & tokenizer persistence, and standalone ST/HF rebuild."""

    logging.info(f"Testing MICE V2 with model_id={model_id}, n_contextualization_layers={n_contextualization_layers}, n_docs_ctx_layers={n_docs_ctx_layers}, bound_bottom_layers={bound_bottom_layers}")

    logging.info("\n\n#1 - Initializing MICE V2 scorer and tasks...")
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

        # 1. Save model and tokenizer
        scorer.save_model(save_path)

        assert (save_path / "config.json").exists(), "config.json missing from save_path!"
        assert (save_path / "model.safetensors").exists() or (save_path / "pytorch_model.bin").exists(), "Weights missing from save_path!"
        assert (save_path / "tokenizer_config.json").exists(), "tokenizer_config.json missing from save_path!"

        # 2. Verify standalone rebuild without base model using HF Auto Classes
        standalone_tok = AutoTokenizer.from_pretrained(save_path)
        assert standalone_tok is not None, "Failed to load tokenizer from save_path!"

        standalone_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        assert standalone_model is not None, "Failed to rebuild model from save_path without base model!"

        # 3. Reload ModernBertMiceScorer
        logging.info("\n\n#3 - Reloading ModernBertMiceScorer...")
        reloaded_scorer = ModernBertMiceScorer.C(
            hf_id=model_id,
            tokenizer=scorer_cfg.tokenizer,
            n_contextualization_layers=n_contextualization_layers,
        ).instance()
        reloaded_scorer.load_model(save_path)

        assert torch.allclose(
            scorer.hf_model.classifier.weight, reloaded_scorer.hf_model.classifier.weight
        ), "Classifier weights mismatch after standard reload!"

        # 4. SentenceTransformers Export & Import Roundtrip
        scorer.export_to_st(str(st_save_path))

        logging.info("\n\n#4 - Loading exported model with SentenceTransformers CrossEncoder...")
        st_cross_encoder = CrossEncoder(str(st_save_path))
        scores = st_cross_encoder.predict([["What is the capital of France?", "Paris is the capital of France."]])
        assert scores is not None
        assert len(scores) == 1


@pytest.mark.parametrize("use_upper_rope", [True, False])
def test_mice_v2_use_upper_rope(use_upper_rope):
    """Tests MICE V2 forward pass and serialization with use_upper_rope enabled vs disabled."""
    model_id = "jhu-clsp/ettin-encoder-17m"
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        use_upper_rope=use_upper_rope,
    )

    test_forward_task = TestMiceV2ForwardTask.C(scorer=scorer_cfg).instance()
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()
    scorer = test_forward_task.scorer

    assert scorer.hf_model.config.use_upper_rope == use_upper_rope

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        scorer.save_model(save_path)

        standalone_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        assert standalone_model.config.use_upper_rope == use_upper_rope
