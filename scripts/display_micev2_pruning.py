#!/usr/bin/env python3
"""
Script for MICE v2 Hub model loading, output parity verification,
and document token utility pruning visualization using circuitsvis.

Usage:
  uv run python scripts/display_micev2_pruning.py \
      [--model-path HF_MODEL_ID_OR_PATH] \
      [--device DEVICE] \
      [--threshold THRESHOLD] \
      [--query QUERY] [--doc DOC] \
      [--output-html HTML_PATH]
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from circuitsvis.tokens import colored_tokens
from sentence_transformers import CrossEncoder
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("display_micev2_pruning")


def analyze_pair_utilities(
    tokenizer: AutoTokenizer,
    doc_ids: torch.Tensor,
    doc_mask: torch.Tensor,
    utilities: torch.Tensor,
    threshold: float,
    pair_index: int = 0,
    query_text: str = "",
    doc_text: str = "",
) -> Dict[str, Any]:
    """
    Formats and prints token-level utility scores and pruning decisions for a pair,
    and returns visualization metadata.
    """
    if doc_ids.ndim > 1:
        doc_ids = doc_ids[pair_index]
    if doc_mask.ndim > 1:
        doc_mask = doc_mask[pair_index]
    if utilities.ndim > 1:
        if utilities.ndim == 3:
            utilities = utilities[pair_index, :, 0]
        else:
            utilities = utilities[pair_index]

    valid_len = int(doc_mask.sum().item())
    doc_ids_valid = doc_ids[:valid_len]
    utils_valid = utilities[:valid_len].cpu().tolist()

    raw_tokens = tokenizer.convert_ids_to_tokens(doc_ids_valid)
    # Replace BPE 'Ġ' with proper space
    tokens = [t.replace("Ġ", " ") for t in raw_tokens]

    statuses = []
    kept_tokens = []
    pruned_tokens = []

    all_below = all(u < threshold for u in utils_valid)
    max_u_idx = int(np.argmax(utils_valid)) if len(utils_valid) > 0 else -1

    for idx, (tok, u) in enumerate(zip(tokens, utils_valid)):
        if u >= threshold or (all_below and idx == max_u_idx):
            status = "KEPT"
            kept_tokens.append(tok)
        else:
            status = "PRUNED"
            pruned_tokens.append(tok)
        statuses.append(status)

    logger.info("=========================================================")
    logger.info("DOCUMENT TOKEN UTILITY PRUNING BREAKDOWN (Pair %d)", pair_index + 1)
    logger.info("Query: '%s'", query_text)
    logger.info("Threshold: %.4f | Total Valid Tokens: %d", threshold, valid_len)
    logger.info("---------------------------------------------------------")
    logger.info(f"{'Idx':<4} | {'Token':<20} | {'Utility (Prob)':<14} | {'Status':<7}")
    logger.info("---------------------------------------------------------")
    for idx, (tok, u, st) in enumerate(zip(tokens, utils_valid, statuses)):
        logger.info(f"{idx:<4} | {tok:<20} | {u:<14.4f} | [{st}]")
    logger.info("---------------------------------------------------------")
    logger.info(
        "Summary: Kept %d tokens (%.1f%%), Pruned %d tokens (%.1f%%)",
        len(kept_tokens),
        (len(kept_tokens) / valid_len * 100) if valid_len > 0 else 0.0,
        len(pruned_tokens),
        (len(pruned_tokens) / valid_len * 100) if valid_len > 0 else 0.0,
    )
    logger.info("=========================================================")

    # Render circuitsvis colored_tokens with raw sigmoid probability utility values
    vis_html_fragment = str(colored_tokens(tokens, utils_valid))

    return {
        "pair_index": pair_index,
        "query": query_text,
        "doc": doc_text,
        "valid_len": valid_len,
        "kept_count": len(kept_tokens),
        "pruned_count": len(pruned_tokens),
        "vis_html_fragment": vis_html_fragment,
    }


def generate_single_html_report(
    pair_results: List[Dict[str, Any]],
    model_name: str,
    threshold: float,
    output_html: str,
):
    """
    Combines all test pair visualizations into a single clean HTML report file.
    Contains page header with metadata, per-document sparsity metrics, and sequential visualization blocks.
    """
    pair_sections = []
    for res in pair_results:
        kept = res["kept_count"]
        pruned = res["pruned_count"]
        total = res["valid_len"]
        kept_pct = (kept / total * 100) if total > 0 else 0.0
        pruned_pct = (pruned / total * 100) if total > 0 else 0.0
        vis_html = res["vis_html_fragment"]

        section = f"""
        <div class="doc-block">
            <p class="stats-line"><strong>Sparsity Metrics:</strong> Kept {kept}/{total} tokens ({kept_pct:.1f}%), Pruned {pruned} tokens ({pruned_pct:.1f}%).</p>
            {vis_html}
        </div>
        """
        pair_sections.append(section)

    combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MICE V2 Token Utility Pruning Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 30px;
            background-color: #ffffff;
            color: #212529;
            line-height: 1.6;
        }}
        h1 {{
            color: #0d6efd;
            border-bottom: 2px solid #0d6efd;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .meta-info {{
            background: #e9ecef;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 25px;
            font-size: 0.95rem;
        }}
        .stats-line {{
            font-size: 0.95rem;
            color: #495057;
            margin-top: 0;
            margin-bottom: 10px;
        }}
        .doc-block {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        /* Hide any internal circuitsvis headers */
        .doc-block header,
        .doc-block h1,
        .doc-block h2,
        .doc-block h3,
        .doc-block h4 {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <h1>MICE V2 Document Token Utility Analysis</h1>
    <div class="meta-info">
        <p style="margin: 3px 0;"><strong>Model:</strong> <code>{model_name}</code></p>
        <p style="margin: 3px 0;"><strong>Pruning Threshold (Sigmoid Prob):</strong> <code>{threshold:.4f}</code></p>
    </div>
    {"".join(pair_sections)}
</body>
</html>
"""
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined_html)
    logger.info("Saved single combined HTML visualization to: %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Load MICE model from Hub/local, verify output parity, and visualize token utility pruning in a single HTML file."
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="VictorMorand/MiceV2-l10-ettin-68m",
        help="Path or HF repo ID of MICE model (default: VictorMorand/MiceV2-l10-ettin-68m).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda', 'mps', or 'cpu').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom utility hard pruning threshold. If omitted, uses model.config.hard_prune_threshold.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional custom query for testing token pruning.",
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Optional custom document for testing token pruning.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="pruning_visualization.html",
        help="Path to save the single combined HTML token visualization (default: pruning_visualization.html).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    target_hf_id = args.model_path
    logger.info("Loading config for model: %s", target_hf_id)
    loaded_cfg = AutoConfig.from_pretrained(target_hf_id, trust_remote_code=True)
    is_mice_model = (
        getattr(loaded_cfg, "model_type", None) == "modernbert_mice"
        or isinstance(loaded_cfg, ModernBertMiceConfig)
    )

    if not is_mice_model:
        raise ValueError(
            f"Config model_type is '{getattr(loaded_cfg, 'model_type', None)}', "
            f"expected a 'modernbert_mice' model!"
        )

    logger.info("#1 - Initializing MICE V2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_hf_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        target_hf_id, trust_remote_code=True
    ).to(device)
    model.eval()

    threshold = (
        args.threshold
        if args.threshold is not None
        else getattr(model.config, "hard_prune_threshold", 0.1)
    )
    logger.info(
        "MICE model initialized. Hard Pruning Enabled: %s, Threshold: %.4f",
        getattr(model.config, "hard_pruning", False),
        threshold,
    )

    # Define test pairs with longer contexts
    if args.query and args.doc:
        test_pairs = [(args.query, args.doc)]
    else:
        test_pairs = [
            (
                "What is Information Retrieval?",
                "Information Retrieval (IR) is the process of obtaining information system resources that are relevant to an information need from a collection of those resources. Searches can be based on full-text or other content-based indexing. Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for metadata that describes data, and for databases of text, imagery or sound. Automated information retrieval systems are used to reduce information overload. An IR system is a software module that provides access to books, articles, or other documents and stores them in an organized structure.",
            ),
            (
                "How does MICE hard token pruning work?",
                "ModernBERT MICE (Modular Independent Contextual Encoding) is a two-stage Cross-Encoder architecture designed to accelerate Information Retrieval. In the first phase, documents and queries are encoded independently using bottom contextualization layers. Next, a lightweight Utility Predictor evaluates every document token's information density and outputs a score between 0 and 1. Upper cross-attention layers then hard prune uninformative document tokens whose predicted utility is below the configured threshold. This drastically reduces upper-layer cross-attention FLOPs, memory bandwidth overhead, and latency while preserving ranking accuracy.",
            ),
        ]

    queries = [q for q, _ in test_pairs]
    docs = [d for _, d in test_pairs]
    logger.info("Running inference on %d test pairs...", len(test_pairs))

    # 2. Compute baseline & independent forward pass logits
    enc_q = tokenizer(queries, padding=True, return_tensors="pt").to(device)
    enc_d = tokenizer(docs, padding=True, return_tensors="pt").to(device)
    pair_enc = tokenizer(queries, docs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        pair_out = model(
            input_ids=pair_enc["input_ids"],
            attention_mask=pair_enc["attention_mask"],
        )
        baseline_pair_scores = pair_out.logits.squeeze(-1).cpu().numpy()

        indep_out = model(
            query_ids=enc_q["input_ids"],
            query_mask=enc_q["attention_mask"],
            doc_ids=enc_d["input_ids"],
            doc_mask=enc_d["attention_mask"],
        )
        baseline_indep_scores = indep_out.logits.squeeze(-1).cpu().numpy()
        utilities_tensor = getattr(indep_out, "utilities", None)

    if utilities_tensor is None:
        raise RuntimeError("Model output did not contain 'utilities'!")

    logger.info("Baseline Pair Scores (ST mode): %s", baseline_pair_scores)
    logger.info("Baseline Independent Scores:  %s", baseline_indep_scores)

    # 3. Test SentenceTransformers CrossEncoder Parity
    logger.info("\n#2 - Testing SentenceTransformers CrossEncoder Parity...")
    st_cross_encoder = CrossEncoder(
        target_hf_id, device=str(device), trust_remote_code=True
    )
    st_scores = np.array(st_cross_encoder.predict(test_pairs))
    expected_st_scores = torch.sigmoid(torch.tensor(baseline_pair_scores)).numpy()

    logger.info("SentenceTransformers CrossEncoder output: %s", st_scores)
    logger.info("Expected ST output (Sigmoid of logits):  %s", expected_st_scores)

    if not np.allclose(expected_st_scores, st_scores, atol=1e-4):
        raise ValueError(
            f"Parity check failed between Sigmoid(baseline) and SentenceTransformers CrossEncoder!\n"
            f"Expected: {expected_st_scores}\nST Output: {st_scores}"
        )
    logger.info("[SUCCESS] Output parity check PASSED!")

    # 4. Analyze token pruning & build combined HTML report
    logger.info("\n#3 - Analyzing Document Token Utilities & Building Combined HTML Visualization...")
    pair_results = []
    for i, (q, d) in enumerate(test_pairs):
        res = analyze_pair_utilities(
            tokenizer=tokenizer,
            doc_ids=enc_d["input_ids"],
            doc_mask=enc_d["attention_mask"],
            utilities=utilities_tensor,
            threshold=threshold,
            pair_index=i,
            query_text=q,
            doc_text=d,
        )
        pair_results.append(res)

    generate_single_html_report(
        pair_results=pair_results,
        model_name=target_hf_id,
        threshold=threshold,
        output_html=args.output_html,
    )

    logger.info("=========================================================")
    logger.info("ALL PARITY CHECKS AND UTILITY VISUALIZATIONS COMPLETED!")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()
