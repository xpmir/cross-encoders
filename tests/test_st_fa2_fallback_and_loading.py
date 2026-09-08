import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from xpmir.neural.huggingface import HFQueryDocTokenizer
from xpmir.neural.sentence_transformers import (
    STCrossEncoder,
    STCrossEncoderModuleLoader,
    st_cross_scorer,
)


def test_hf_query_doc_tokenizer_fa2_fallback():
    """Verify that HFQueryDocTokenizer.should_flatten() returns False when FA2 is unavailable on hardware."""
    tokenizer = HFQueryDocTokenizer.C(
        model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length=512,
        max_query_length=64,
        max_doc_length=448,
        pref_attn_implementation="flash_attention_2",
    ).instance()

    # When is_fa2_available returns False (e.g. V100 GPU or CPU)
    with patch("xpm_torch.utils.fabric.is_fa2_available", return_value=False):
        assert tokenizer.should_flatten() is False

    # When is_fa2_available returns True, and transformers reports varlen is available
    with patch("xpm_torch.utils.fabric.is_fa2_available", return_value=True):
        with patch("transformers.utils.generic.is_flash_attention_requested", return_value=True):
            with patch(
                "transformers.modeling_flash_attention_utils.lazy_import_flash_attention",
                return_value=((None, MagicMock(), None), None),
            ):
                assert tokenizer.should_flatten() is True


def test_st_cross_encoder_fa2_fallback(caplog):
    """Verify that STCrossEncoder falls back from FA2 to SDPA when hardware does not support FA2."""
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"

    cfg = STCrossEncoder.C(
        model_id=model_id,
        pref_attn_implementation="flash_attention_2",
    )
    scorer = cfg.instance()

    with patch("xpm_torch.utils.fabric.is_fa2_available", return_value=False):
        import logging
        with caplog.at_level(logging.WARNING):
            scorer.initialize()

        assert "Flash Attention 2 requested" in caplog.text
        assert "Falling back to `_attn_implementation='sdpa'`" in caplog.text
        assert scorer._initialized is True


def test_st_cross_encoder_module_loader_prevents_backbone_reload(tmp_path):
    """Verify that STCrossEncoderModuleLoader loads checkpoint directly and initialize() is a no-op."""
    # 1. First save a valid model to a temporary directory
    base_model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    init_scorer = STCrossEncoder.C(model_id=base_model_id).instance()
    init_scorer.initialize()

    ckpt_dir = tmp_path / "checkpoint"
    init_scorer.save_model(ckpt_dir)

    # Verify modules.json exists in ckpt_dir
    assert (ckpt_dir / "modules.json").is_file()

    # 2. Create a new STCrossEncoder with an INVALID base model_id and explicit max_length
    # If initialize() were called before load_model, this would raise an error (or attempt to download)
    dummy_model_id = "nonexistent-org/invalid-model-name-12345"
    eval_cfg = STCrossEncoder.C(model_id=dummy_model_id, max_length=512)

    # Obtain the loader_config
    loader_cfg = eval_cfg.loader_config(ckpt_dir)
    loader = loader_cfg.instance()
    assert isinstance(loader, STCrossEncoderModuleLoader)

    # Execute the loader
    loader.execute()

    eval_scorer = loader.value
    # Assert model is loaded and initialized from the checkpoint, NOT dummy_model_id
    assert eval_scorer._initialized is True
    assert eval_scorer.model_id == str(ckpt_dir)
    assert eval_scorer.st_model is not None
