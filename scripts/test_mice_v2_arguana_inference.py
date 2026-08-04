#!/usr/bin/env python3
"""
ArguAna Real-Batch Inference & Parity Benchmark Script for MICE v2 (SDPA vs FA2).

Loads real query-document pairs from datamaestro ('org.beir.arguana' dataset),
logs model and RoPE parameters, and measures inference relevance scores and numerical differences
between SDPA and FA2 implementations.

Usage:
    uv run python scripts/test_mice_v2_arguana_inference.py [--device cuda] [--dtype bfloat16] [--model-path PATH_OR_HF_ID] [--num-samples 8]
"""

import argparse
import logging
import torch
from datamaestro import prepare_dataset
from transformers import AutoTokenizer

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mice_v2_arguana_inference")


def log_model_rope_info(config: ModernBertMiceConfig, model_name: str = "Model"):
    """Logs RoPE parameters, layer types, and attention implementation details."""
    logger.info("=" * 80)
    logger.info(f"MODEL & RoPE CONFIGURATION SUMMARY ({model_name})")
    logger.info("=" * 80)
    logger.info(f"Attention Implementation: '{getattr(config, '_attn_implementation', 'sdpa')}'")
    logger.info(f"Num Hidden Layers: {getattr(config, 'num_hidden_layers', None)}")
    logger.info(f"Contextualization Layers: {getattr(config, 'n_contextualization_layers', None)}")
    logger.info(f"Interaction Layers: {getattr(config, 'n_interaction_layers', None)}")
    logger.info(f"Local Attention Window: {getattr(config, 'local_attention', None)}")
    logger.info(f"RoPE Parameters: {getattr(config, 'rope_parameters', None)}")
    layer_types = getattr(config, "layer_types", [])
    logger.info(f"Layer Types ({len(layer_types)} total): {layer_types[:8]} ...")
    logger.info("=" * 80)


def load_tokenizer_for_model(model_path: str = None, fallback_id: str = "jhu-clsp/ettin-encoder-68m"):
    """Loads tokenizer from model_path or falls back to base model tokenizer if missing in checkpoint dir."""
    if model_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded tokenizer directly from model_path: '{model_path}'")
            return tokenizer
        except Exception as e:
            logger.warning(f"Could not load tokenizer directly from '{model_path}': {e}. Trying fallback tokenizer...")

        try:
            cfg = ModernBertMiceConfig.from_pretrained(model_path)
            base_id = getattr(cfg, "_name_or_path", fallback_id)
            if base_id and base_id != model_path:
                logger.info(f"Loading tokenizer from config _name_or_path: '{base_id}'")
                return AutoTokenizer.from_pretrained(base_id)
        except Exception:
            pass

    logger.info(f"Loading fallback tokenizer: '{fallback_id}'")
    return AutoTokenizer.from_pretrained(fallback_id)


def load_arguana_datamaestro_batch(
    num_samples: int = 8, model_path: str = None, fallback_tokenizer_id: str = "jhu-clsp/ettin-encoder-68m"
):
    """Loads num_samples real query-document pairs from datamaestro 'org.beir.arguana'."""
    logger.info(f"Loading datamaestro dataset 'org.beir.arguana' ({num_samples} samples)...")
    ds = prepare_dataset("org.beir.arguana").instance()

    topic_map = {t["id"]: t["text_item"].text for t in ds.topics.iter()}

    pairs = []
    for qrel in ds.assessments.iter():
        q_text = topic_map[qrel.topic_id]
        for assessment in qrel.assessments:
            doc = ds.documents.document_ext(assessment.doc_id)
            d_text = doc["text_item"].text
            pairs.append((q_text, d_text))
            if len(pairs) >= num_samples:
                break
        if len(pairs) >= num_samples:
            break

    tokenizer = load_tokenizer_for_model(model_path=model_path, fallback_id=fallback_tokenizer_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", 50283)

    q_ids_list, d_ids_list = [], []
    q_mask_list, d_mask_list = [], []
    q_lens, d_lens = [], []

    for q_text, d_text in pairs:
        q_enc = tokenizer(q_text, truncation=True, max_length=512, return_tensors="pt")
        d_enc = tokenizer(d_text, truncation=True, max_length=512, return_tensors="pt")

        q_ids = q_enc["input_ids"].squeeze(0)
        q_mask = q_enc["attention_mask"].squeeze(0)
        d_ids = d_enc["input_ids"].squeeze(0)
        d_mask = d_enc["attention_mask"].squeeze(0)

        q_ids_list.append(q_ids)
        q_mask_list.append(q_mask)
        d_ids_list.append(d_ids)
        d_mask_list.append(d_mask)

        q_lens.append(len(q_ids))
        d_lens.append(len(d_ids))

    logger.info(
        f"Datamaestro ArguAna sample lengths -> Query avg: {sum(q_lens)/len(q_lens):.1f} toks (max {max(q_lens)}), "
        f"Doc avg: {sum(d_lens)/len(d_lens):.1f} toks (max {max(d_lens)})"
    )

    pad_id = tokenizer.pad_token_id
    query_ids = torch.nn.utils.rnn.pad_sequence(q_ids_list, batch_first=True, padding_value=pad_id)
    query_mask = torch.nn.utils.rnn.pad_sequence(q_mask_list, batch_first=True, padding_value=0)
    doc_ids = torch.nn.utils.rnn.pad_sequence(d_ids_list, batch_first=True, padding_value=pad_id)
    doc_mask = torch.nn.utils.rnn.pad_sequence(d_mask_list, batch_first=True, padding_value=0)

    return query_ids, query_mask, doc_ids, doc_mask


def test_arguana_inference(
    device: torch.device, dtype: torch.dtype, model_path: str = None, num_samples: int = 8
):
    if model_path:
        logger.info(f"Loading pretrained model config from: '{model_path}'")
        base_config = ModernBertMiceConfig.from_pretrained(model_path)
    else:
        base_config = ModernBertMiceConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=8,
            n_contextualization_layers=4,
            n_interaction_layers=4,
            pad_token_id=50283,
            sep_token_id=50282,
            cls_token_id=50281,
            local_attention=128,
        )

    log_model_rope_info(base_config, model_name=model_path or "Synthetic Base Config")

    query_ids, query_mask, doc_ids, doc_mask = load_arguana_datamaestro_batch(
        num_samples=num_samples, model_path=model_path
    )

    query_ids = query_ids.to(device)
    query_mask = query_mask.to(device)
    doc_ids = doc_ids.to(device)
    doc_mask = doc_mask.to(device)

    config_sdpa = ModernBertMiceConfig(**base_config.to_dict())
    config_sdpa._attn_implementation = "sdpa"

    config_fa2 = ModernBertMiceConfig(**base_config.to_dict())
    config_fa2._attn_implementation = "flash_attention_2"

    logger.info("Initializing SDPA and FA2 models...")
    if model_path:
        logger.info(f"Loading weights from checkpoint '{model_path}' for SDPA & FA2 models...")
        model_sdpa = ModernBertMiceForSequenceClassification.from_pretrained(
            model_path, config=config_sdpa
        ).to(device=device, dtype=dtype)
        model_fa2 = ModernBertMiceForSequenceClassification.from_pretrained(
            model_path, config=config_fa2
        ).to(device=device, dtype=dtype)
    else:
        torch.manual_seed(42)
        model_sdpa = ModernBertMiceForSequenceClassification(config_sdpa).to(device=device, dtype=dtype)
        torch.manual_seed(42)
        model_fa2 = ModernBertMiceForSequenceClassification(config_fa2).to(device=device, dtype=dtype)
        model_fa2.load_state_dict(model_sdpa.state_dict())

    model_sdpa.eval()
    model_fa2.eval()

    logger.info("Running forward inference pass on Datamaestro ArguAna batch...")
    with torch.no_grad():
        out_sdpa = model_sdpa(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)
        out_fa2 = model_fa2(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    logits_sdpa = out_sdpa.logits.squeeze(-1)
    logits_fa2 = out_fa2.logits.squeeze(-1)

    diff = (logits_sdpa - logits_fa2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info("-" * 80)
    logger.info("ARGUANA DATAMAESTRO REAL-BATCH SCORE COMPARISON (SDPA vs FA2)")
    logger.info("-" * 80)
    for i in range(min(num_samples, 5)):
        logger.info(f"Sample {i+1}: SDPA Score = {logits_sdpa[i].item():.6f} | FA2 Score = {logits_fa2[i].item():.6f} | Abs Diff = {diff[i].item():.6f}")

    logger.info("-" * 80)
    logger.info(f"Summary -> Max Abs Diff: {max_diff:.6f} | Mean Abs Diff: {mean_diff:.6f}")
    logger.info("-" * 80)

    return max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser(description="ArguAna Datamaestro Real-Batch SDPA vs FA2 Inference Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    test_arguana_inference(device=device, dtype=dtype, model_path=args.model_path, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
