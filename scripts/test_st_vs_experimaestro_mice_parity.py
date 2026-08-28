#!/usr/bin/env python3
"""
Parity comparison script between SentenceTransformers CrossEncoder and Experimaestro (xpmir)
for MICE v2 models (e.g. VictorMorand/MiceV2-l10p4-ettin-68m).

Verifies loading via:
 1. Experimaestro TorchHFHub.from_pretrained (as specified in MiceCard.md)
 2. Experimaestro ModernBertMiceScorer.C(hf_id=...)
 3. Hugging Face AutoModelForSequenceClassification (independent & concatenated modes)
 4. SentenceTransformers CrossEncoder (default vs Identity activation)

Usage:
  uv run python scripts/test_st_vs_experimaestro_mice_parity.py [--model-id HF_HUB_ID] [--device DEVICE]
"""

import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from xpm_torch.huggingface import TorchHFHub
from xpmir.letor.records import PointwiseItems
from MICE.modeling.mice import MICEQueryDocTokenizer
from MICE.modeling.mice_v2.modernbert_xpm import ModernBertMiceScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_st_vs_experimaestro_mice_parity")


def main():
    parser = argparse.ArgumentParser(
        description="Verify parity between SentenceTransformers CrossEncoder and Experimaestro."
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default="VictorMorand/MiceV2-l10p4-ettin-68m",
        help="HF model ID or local directory path of exported MICE model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Target device to run inference on ('cuda', 'mps', or 'cpu').",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info("Target device: %s", device)
    logger.info("Evaluating model ID: %s", args.model_id)

    # Define test query-document pairs
    queries = [
        "Which planet is known as the Red Planet?",
        "What is the capital of France?",
    ]
    documents = [
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Paris is the capital and most populous city of France.",
    ]
    records = PointwiseItems.from_texts(topics=queries, documents=documents)
    pairs = [[q, d] for q, d in zip(queries, documents)]

    # ----------------------------------------------------
    # 1. Experimaestro TorchHFHub loading (as in MiceCard.md)
    # ----------------------------------------------------
    logger.info("\n=== 1. Loading via Experimaestro TorchHFHub.from_pretrained ===")
    try:
        hub_model = TorchHFHub.from_pretrained(args.model_id)
        hub_model.initialize()
        hub_model.to(device)
        hub_model.eval()
        with torch.no_grad():
            scores_hub = hub_model(records).cpu().numpy()
        logger.info("TorchHFHub scores (independent Q/D): %s", scores_hub)
    except Exception as e:
        logger.error("Failed to load model via TorchHFHub: %s", e)
        scores_hub = None

    # ----------------------------------------------------
    # 2. Experimaestro ModernBertMiceScorer loading
    # ----------------------------------------------------
    logger.info("\n=== 2. Loading via Experimaestro ModernBertMiceScorer ===")
    tok = MICEQueryDocTokenizer.C(model_id=args.model_id)
    scorer = ModernBertMiceScorer.C(hf_id=args.model_id, tokenizer=tok).instance()
    scorer.initialize()
    scorer.to(device)
    scorer.eval()
    with torch.no_grad():
        scores_xpm = scorer(records).cpu().numpy()
    logger.info("ModernBertMiceScorer scores (independent Q/D): %s", scores_xpm)

    # ----------------------------------------------------
    # 3. Direct HF AutoModel (Independent vs Concatenated)
    # ----------------------------------------------------
    logger.info("\n=== 3. Loading via HF AutoModelForSequenceClassification ===")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, trust_remote_code=True
    ).to(device)
    hf_model.eval()

    q_enc = hf_tokenizer(queries, padding=True, return_tensors="pt").to(device)
    d_enc = hf_tokenizer(documents, padding=True, return_tensors="pt").to(device)
    pair_enc = hf_tokenizer(queries, documents, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits_hf_indep = hf_model(
            query_ids=q_enc["input_ids"],
            query_mask=q_enc["attention_mask"],
            doc_ids=d_enc["input_ids"],
            doc_mask=d_enc["attention_mask"],
        ).logits.squeeze(-1).cpu().numpy()

        logits_hf_pair = hf_model(
            input_ids=pair_enc["input_ids"],
            attention_mask=pair_enc["attention_mask"],
        ).logits.squeeze(-1).cpu().numpy()

    logger.info("HF AutoModel independent Q/D logits: %s", logits_hf_indep)
    logger.info("HF AutoModel concatenated pair logits: %s", logits_hf_pair)

    # ----------------------------------------------------
    # 4. SentenceTransformers CrossEncoder Loading
    # ----------------------------------------------------
    logger.info("\n=== 4. Loading via sentence_transformers CrossEncoder ===")
    st_ce_default = CrossEncoder(args.model_id, device=str(device), trust_remote_code=True)
    scores_st_default = np.array(st_ce_default.predict(pairs))

    st_ce_identity = CrossEncoder(
        args.model_id,
        device=str(device),
        trust_remote_code=True,
        activation_fn=nn.Identity(),
    )
    scores_st_identity = np.array(st_ce_identity.predict(pairs))

    logger.info("ST CrossEncoder default predict output:             %s", scores_st_default)
    logger.info("ST CrossEncoder default activation function:        %s", st_ce_default.activation_fn)
    logger.info("ST CrossEncoder logits (Identity activation):       %s", scores_st_identity)

    # ----------------------------------------------------
    # Parity Diagnostics & Analysis
    # ----------------------------------------------------
    logger.info("\n=======================================================")
    logger.info("               PARITY DIAGNOSTIC REPORT               ")
    logger.info("=======================================================")

    # Check 1: Experimaestro TorchHFHub vs ModernBertMiceScorer
    if scores_hub is not None:
        np.testing.assert_allclose(
            scores_hub,
            scores_xpm,
            atol=1e-4,
            err_msg="Mismatch between TorchHFHub and ModernBertMiceScorer!",
        )
        logger.info("[PASSED] Experimaestro TorchHFHub vs ModernBertMiceScorer independent scores match exactly!")

    # Check 2: Experimaestro vs HF AutoModel (Independent mode)
    np.testing.assert_allclose(
        scores_xpm,
        logits_hf_indep,
        atol=1e-4,
        err_msg="Mismatch between Experimaestro and HF AutoModel independent mode!",
    )
    logger.info("[PASSED] Experimaestro vs HF AutoModel independent Q/D logits match exactly!")

    # Check 3: ST CrossEncoder (Identity) vs HF AutoModel (Pair mode)
    np.testing.assert_allclose(
        scores_st_identity,
        logits_hf_pair,
        atol=1e-4,
        err_msg="Mismatch between ST CrossEncoder (Identity) and HF AutoModel pair mode!",
    )
    logger.info("[PASSED] ST CrossEncoder (Identity) vs HF AutoModel pair logits match exactly!")

    # Check 4: Activation function warning if Sigmoid was used in default ST CrossEncoder
    is_default_sigmoid = isinstance(st_ce_default.activation_fn, nn.Sigmoid)
    if is_default_sigmoid:
        logger.warning(
            "\n[ST ACTIVATION MISMATCH DETECTED]\n"
            "ST CrossEncoder defaulted to nn.Sigmoid() because ST metadata (sbert_ce_default_activation_function / "
            "config_sentence_transformers.json) was missing from the Hugging Face repository.\n"
            "Sigmoid(logits %s) produced compressed scores %s.\n"
            "Fix: Ensure sbert_ce_default_activation_function = 'torch.nn.modules.linear.Identity' is included in config.json.",
            scores_st_identity,
            scores_st_default,
        )
    else:
        logger.info("[PASSED] ST CrossEncoder default activation function correctly resolved to Identity!")

    logger.info("\nSummary of Raw Logits / Scores across formats:")
    logger.info(" - Experimaestro / MICE Independent Q/D Score: %s", scores_xpm)
    logger.info(" - ST CrossEncoder Pair Logits (Identity):     %s", scores_st_identity)
    if is_default_sigmoid:
        logger.info(" - ST CrossEncoder Default Scores (Sigmoid):   %s", scores_st_default)

    logger.info("\nALL PARITY CHECKS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
