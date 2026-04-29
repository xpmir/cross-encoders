"""
Tests for the MICE model.
run manually with `pytest tests/test_mice.py` or `uv run pytest tests/test_mice.py -v`
"""

import torch
import logging
import tempfile
import pytest
from pathlib import Path
from MICE.modeling.mice import mice_scorer, MiceCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param
from xpm_torch.huggingface import TorchHFHub


class TestMiceForwardTask(LightweightTask):
    """Tests Mice Loading, scoring a PointWiseItem, and compare outputs after reloading for checkpointing"""

    scorer: Param[MiceCrossEncoder]

    def execute(self):
        """
        Task execution for instantiating a dataset and running a forward pass for a MICE model.
        """
        print(f"Step 1: Instantiating the MICE model (ID: {id(self.scorer)})...")

        # Initialize the model components (experimaestro way)
        self.scorer.initialize()

        print("Step 2: Preparing the dataset...")
        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]

        # Create the input records
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        print("Step 3: Running a forward pass...")
        self.scorer.eval()

        # Run forward pass without gradient computation
        with torch.no_grad():
            output = self.scorer(input_records)

        print(f"Model output (score): {output}")

        # Validation
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"

        print("\nForward pass successful!")


@pytest.mark.parametrize(
    "model_id",
    [
        "jhu-clsp/ettin-encoder-68m",  # ModernBERT
        # "cross-encoder/ms-marco-MiniLM-L-6-v2",  # BERT
        # "Qwen/Qwen2.5-0.5B-Instruct",           # Qwen
    ],
)
@pytest.mark.parametrize("cross_attn_first", [True, False])
@pytest.mark.parametrize("mask_cls_to_doc", [False])
@pytest.mark.parametrize("n_contextualization_layers", [3])
def test_mice(
    model_id,
    n_contextualization_layers,
    cross_attn_first: bool,
    mask_cls_to_doc: bool,
):
    """Tests MICE loading, forward pass, and weight persistence across a grid of parameters."""

    # Initialize scorer configuration
    scorer_cfg, init_tasks = mice_scorer(
        hf_id=model_id,
        n_contextualization_layers=n_contextualization_layers,
        cross_attn_first=cross_attn_first,
        mask_cls_to_doc=mask_cls_to_doc,
    )

    # Create and run the forward pass task
    test_forward_task = TestMiceForwardTask.C(scorer=scorer_cfg).instance()

    # Run initialization tasks (weight loading)
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()

    # --- Persistence Verification ---
    model = test_forward_task.scorer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"
        hf_save_path = tmp_path / "hf_model"

        # 1. Save model weights
        model.save_model(save_path)

        # 2. Reload model using loader_config
        loader_config = scorer_cfg.loader_config(save_path)
        loader = loader_config.instance()
        loader.execute()
        reloaded_model = loader.model

        # 3. Verify weights are identical
        assert torch.allclose(
            model.classifier.weight, reloaded_model.classifier.weight
        ), "Classifier weights mismatch after standard reload!"

        # 4. HF Export/Import Roundtrip
        hub = TorchHFHub(loader_config)
        hub.save_pretrained(hf_save_path)

        hf_loader = TorchHFHub.pretrained_loader(hf_save_path, as_instance=True)
        hf_loader.execute()
        hf_model = hf_loader.model

        assert torch.allclose(
            model.classifier.weight, hf_model.classifier.weight
        ), "Classifier weights mismatch after HF roundtrip!"
