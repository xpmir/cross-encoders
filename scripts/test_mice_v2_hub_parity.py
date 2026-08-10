#!/usr/bin/env python3
"""
Test script for MICE v2 Hub publishing, reloading, and output parity.
Verifies output consistency across:
 1. Initial MICE v2 model
 2. Local exported model loaded via HF AutoModelForSequenceClassification
 3. Local exported model loaded via SentenceTransformers CrossEncoder
 4. (Optional) Remote model pushed to and reloaded from Hugging Face Hub

Usage:
  uv run python scripts/test_mice_v2_hub_parity.py [--model-path PATH_OR_HF_ID] [--repo-id HF_REPO_ID] [--device DEVICE]
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from MICE.modeling.mice import MICEQueryDocTokenizer
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    ModernBertMiceScorer,
    mice_v2_scorer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mice_v2_hub_parity")


def main():
    parser = argparse.ArgumentParser(
        description="Test MICE v2 model export, HF Hub push/reload, and output numerical parity."
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=None,
        help="Optional path to a local MICE v2 model directory or HF model ID. If omitted, a test model is created.",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        type=str,
        default=None,
        help="Optional target HF Hub repository ID (e.g. 'username/mice-v2-test') to test pushing and remote reloading.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory to save exported model artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device to run inference on ('cuda', 'mps', or 'cpu').",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    target_hf_id = args.model_path or "jhu-clsp/ettin-encoder-17m"
    loaded_cfg = AutoConfig.from_pretrained(target_hf_id)
    is_mice_model = (
        getattr(loaded_cfg, "model_type", None) == "modernbert_mice"
        or isinstance(loaded_cfg, ModernBertMiceConfig)
    )

    logger.info("\n\n#1 - Initializing MICE V2 scorer and tasks...")

    if is_mice_model:
        logger.info("Provided path/id '%s' is an existing MICE model checkpoint.", target_hf_id)
        tokenizer = MICEQueryDocTokenizer.C(model_id=target_hf_id)
        scorer = ModernBertMiceScorer.C(
            hf_id=target_hf_id,
            tokenizer=tokenizer,
            n_contextualization_layers=getattr(loaded_cfg, "n_contextualization_layers", 2),
        ).instance()
        scorer.initialize()
    else:
        logger.info(
            "Path/id '%s' is a raw backbone. Initializing MICE model using InitMICEV2ModernBERTFromHFID...",
            target_hf_id,
        )
        scorer, init_tasks = mice_v2_scorer(
            hf_id=target_hf_id,
            n_contextualization_layers=2,
            n_interaction_layers=4,
            bound_bottom_layers=True,
        )
        scorer = scorer.instance()
        for init_task in init_tasks:
            logger.info("Executing initialization task: %s", init_task)
            init_task.instance().execute()
        scorer.initialize()

    model = scorer.hf_model
    model.to(device)
    model.eval()

    tokenizer = scorer.tokenizer.tokenizer

    # Define test query-document pairs
    test_pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("How does MICE work?", "MICE encodes queries and documents independently before cross-attending."),
    ]
    logger.info("Test pairs: %s", test_pairs)

    # 2. Compute baseline scores for pair tokenization & independent tokenization
    logger.info("Computing baseline scores for pair tokenization & independent tokenization...")
    queries = [q for q, _ in test_pairs]
    docs = [d for _, d in test_pairs]

    pair_enc = tokenizer(queries, docs, padding=True, return_tensors="pt").to(device)
    enc_q = tokenizer(queries, padding=True, return_tensors="pt").to(device)
    enc_d = tokenizer(docs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        # Baseline via concatenated pair input (SentenceTransformers mode)
        pair_out = model(
            input_ids=pair_enc["input_ids"],
            attention_mask=pair_enc["attention_mask"],
        )
        baseline_pair_scores = pair_out.logits.squeeze(-1).cpu().numpy()

        # Baseline via independent query/doc input
        indep_out = model(
            query_ids=enc_q["input_ids"],
            query_mask=enc_q["attention_mask"],
            doc_ids=enc_d["input_ids"],
            doc_mask=enc_d["attention_mask"],
        )
        baseline_indep_scores = indep_out.logits.squeeze(-1).cpu().numpy()

    logger.info("Baseline Pair Scores (ST mode): %s", baseline_pair_scores)
    logger.info("Baseline Independent Scores:  %s", baseline_indep_scores)

    # 3. Export model locally using scorer.export_to_st
    if args.temp_dir:
        export_dir = Path(args.temp_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory()
        export_dir = Path(tmp_context.name)

    logger.info("Exporting model using scorer.export_to_st to: %s", export_dir)
    scorer.export_to_st(str(export_dir), repo_id=args.repo_id)

    # Check auto_map in saved config.json
    config_json_path = export_dir / "config.json"
    if not config_json_path.exists():
        raise FileNotFoundError(f"config.json not found in export directory: {export_dir}")

    config_content = config_json_path.read_text()
    logger.info("Verifying config.json contents...")
    if '"auto_map"' not in config_content:
        logger.warning("'auto_map' missing in saved config.json!")
    else:
        logger.info("Successfully verified 'auto_map' in saved config.json.")

    # Check README.md model card generated from MiceCard.md
    readme_path = export_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md model card not found in export directory: {export_dir}")
    logger.info("Successfully verified README.md model card generated from MiceCard.md.")

    # 4. Reload via HF AutoModelForSequenceClassification
    logger.info("\n\n#4 - Reloading local model via HF AutoModelForSequenceClassification...")
    logger.info("Reloading local model via AutoModelForSequenceClassification...")
    hf_reloaded = AutoModelForSequenceClassification.from_pretrained(
        export_dir, trust_remote_code=True
    ).to(device)
    hf_reloaded.eval()

    with torch.no_grad():
        hf_pair_out = hf_reloaded(
            input_ids=pair_enc["input_ids"],
            attention_mask=pair_enc["attention_mask"],
        )
        hf_pair_scores = hf_pair_out.logits.squeeze(-1).cpu().numpy()

        hf_indep_out = hf_reloaded(
            query_ids=enc_q["input_ids"],
            query_mask=enc_q["attention_mask"],
            doc_ids=enc_d["input_ids"],
            doc_mask=enc_d["attention_mask"],
        )
        hf_indep_scores = hf_indep_out.logits.squeeze(-1).cpu().numpy()

    logger.info("HF AutoModel Pair reloaded scores:  %s", hf_pair_scores)
    logger.info("HF AutoModel Indep reloaded scores: %s", hf_indep_scores)
    if not np.allclose(baseline_pair_scores, hf_pair_scores, atol=1e-5):
        raise ValueError(
            f"Pair output parity check failed between baseline and HF reloaded model!\n"
            f"Baseline: {baseline_pair_scores}\nHF Reloaded: {hf_pair_scores}"
        )
    if not np.allclose(baseline_indep_scores, hf_indep_scores, atol=1e-5):
        raise ValueError(
            f"Independent output parity check failed between baseline and HF reloaded model!\n"
            f"Baseline: {baseline_indep_scores}\nHF Reloaded: {hf_indep_scores}"
        )
    logger.info("[SUCCESS] Baseline vs. HF AutoModel parity check PASSED!")

    # 5. Reload via SentenceTransformers CrossEncoder
    logger.info("\n\n#5 - Loading exported model with SentenceTransformers CrossEncoder...")
    logger.info("Reloading local model via sentence_transformers.CrossEncoder...")
    st_cross_encoder = CrossEncoder(
        str(export_dir), device=str(device), trust_remote_code=True
    )
    st_scores = np.array(st_cross_encoder.predict(test_pairs))
    expected_st_scores = torch.sigmoid(torch.tensor(baseline_pair_scores)).numpy()

    logger.info("SentenceTransformers CrossEncoder raw output:     %s", st_scores)
    logger.info("Expected ST output (Sigmoid of baseline logits): %s", expected_st_scores)

    if not np.allclose(expected_st_scores, st_scores, atol=1e-5):
        raise ValueError(
            f"Output parity check failed between Sigmoid(baseline) and SentenceTransformers CrossEncoder!\n"
            f"Expected: {expected_st_scores}\nST CrossEncoder: {st_scores}"
        )
    logger.info("[SUCCESS] Baseline Pair (Sigmoid) vs. SentenceTransformers CrossEncoder parity check PASSED!")

    # 6. Optional Hub Push & Remote Reloading
    if args.repo_id:
        logger.info("Pushing model and tokenizer to Hugging Face Hub repo: %s...", args.repo_id)
        model.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)
        logger.info("Successfully pushed model and tokenizer to %s", args.repo_id)

        logger.info("\n\n#6 - Testing remote reload via sentence_transformers.CrossEncoder('%s', trust_remote_code=True)...", args.repo_id)
        hub_cross_encoder = CrossEncoder(
            args.repo_id, device=str(device), trust_remote_code=True
        )
        hub_scores = np.array(hub_cross_encoder.predict(test_pairs))
        logger.info("Hub CrossEncoder remote scores: %s", hub_scores)

        if not np.allclose(expected_st_scores, hub_scores, atol=1e-5):
            raise ValueError(
                f"Output parity check failed between Sigmoid(baseline) and Hub CrossEncoder!\n"
                f"Expected: {expected_st_scores}\nHub CrossEncoder: {hub_scores}"
            )
        logger.info("[SUCCESS] Baseline Pair (Sigmoid) vs. HF Hub remote CrossEncoder parity check PASSED!")

    logger.info("=========================================================")
    logger.info("ALL PARITY CHECKS PASSED SUCCESSFULLY!")
    logger.info("=========================================================")

    if tmp_context is not None:
        tmp_context.cleanup()


if __name__ == "__main__":
    main()
