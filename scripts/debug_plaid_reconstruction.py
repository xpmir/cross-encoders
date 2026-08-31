#!/usr/bin/env python3
"""
PLAID Reconstruction Diagnostic Script for MICE v2.

Localizes why reranking with PLAID-reconstructed document token embeddings gives
much worse results than reranking with document embeddings computed online (no
PLAID). MICE's PLAID index is used purely as a compressed embedding store (built
with `compress_only=True`): documents are found by a first-stage SPLADE retriever,
and their bottom-contextualization-layer token vectors are looked up by document ID
from the index (`PlaidIndex.get_document_tokens(doc_id)`) instead of being
recomputed online. There is no ANN/vector search against the index.

Runs 4 staged checks against a real, already-built PLAID index and a real trained
checkpoint (both supplied by the user -- this script does NOT build a fresh index):

  Stage A: Index metadata sanity (corpus size / hidden-dim match between the index
           and the checkpoint/dataset actually being evaluated).
  Stage B: Per-document reconstruction fidelity -- compares the "exact" bottom-layer
           token embeddings (recomputed online) against the embeddings reconstructed
           from the PLAID index for the SAME real documents (cosine similarity,
           relative L2 error, norm ratio).
  Stage C: Downstream score sensitivity -- feeds (a) online doc_ids, (b) exact
           uncompressed doc_hidden, and (c) PLAID-reconstructed doc_hidden into the
           same MICE forward pass for real (query, relevant-doc) pairs, and reports
           where the 3 scores diverge. This pinpoints whether the bug is in the
           `doc_hidden` bypass code path itself (a vs b already differ), or in PLAID
           compression fidelity (b vs c differ despite a vs b matching).
  Stage D: Full document-ID coverage check between the index's ext2int map and the
           real corpus (catches ID-format/staleness bugs independent of compression
           math).

Hardware: needs the real trained checkpoint's attention implementation (often
FlashAttention-2), so run this on a GPU node (e.g. Jean Zay). Falls back to
whatever `_attn_implementation` is baked into the checkpoint config unless
overridden with --attn-implementation.

Usage:
  uv run python scripts/debug_plaid_reconstruction.py \
      --model-path /path/to/checkpoint \
      --plaid-index-path /path/to/plaid-index-dir \
      --dataset org.beir.nfcorpus.test \
      --num-docs 50 --num-queries 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datamaestro import prepare_dataset

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from transformers import AutoTokenizer
from xpmir.index.plaid import PlaidIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_plaid_reconstruction")


# ---------------------------------------------------------------------------
# Loading helpers (mirrors scripts/test_mice_v2_arguana_inference.py conventions)
# ---------------------------------------------------------------------------


def load_tokenizer_for_model(
    model_path: str, fallback_id: str = "jhu-clsp/ettin-encoder-68m"
):
    """Loads tokenizer from model_path or falls back to base model tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Loaded tokenizer directly from model_path: '{model_path}'")
        return tokenizer
    except Exception as e:
        logger.warning(
            f"Could not load tokenizer directly from '{model_path}': {e}. "
            "Trying fallback tokenizer..."
        )

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


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: Optional[str] = None,
) -> ModernBertMiceForSequenceClassification:
    logger.info(f"Loading MICE v2 checkpoint from '{model_path}'...")
    config = ModernBertMiceConfig.from_pretrained(model_path)
    if attn_implementation:
        config._attn_implementation = attn_implementation
    logger.info(
        f"Model config: hidden_size={config.hidden_size}, "
        f"n_contextualization_layers={getattr(config, 'n_contextualization_layers', None)}, "
        f"n_interaction_layers={getattr(config, 'n_interaction_layers', None)}, "
        f"utility_pred={getattr(config, 'utility_pred', False)}, "
        f"attn_implementation={getattr(config, '_attn_implementation', 'sdpa')}"
    )
    if getattr(config, "utility_pred", False):
        logger.warning(
            "utility_pred=True on this checkpoint -- this script does NOT check "
            "the (separate, known) utility-gating double-application code path. "
            "Results below only isolate PLAID compression/reconstruction fidelity."
        )
    model = ModernBertMiceForSequenceClassification.from_pretrained(
        model_path, config=config
    ).to(device=device, dtype=dtype)
    model.eval()
    return model


def load_dataset(dataset_id: str):
    """Returns (dataset_config, dataset_instance).

    `dataset_config` (un-instantiated) is what must be passed into other
    experimaestro Configs' Param slots (e.g. `PlaidIndex.C(documents=...)`) --
    experimaestro raises `AttributeError: Configuration (and not objects) should
    be used` if you pass an already-`.instance()`-d object there instead.
    `dataset_instance` is the actual Python object used for direct text/iteration
    access (`.documents.iter_documents()`, `.topics.iter()`, etc).
    """
    logger.info(f"Loading datamaestro dataset '{dataset_id}'...")
    dataset_config = prepare_dataset(dataset_id)
    return dataset_config, dataset_config.instance()


# ---------------------------------------------------------------------------
# Stage A: index metadata sanity
# ---------------------------------------------------------------------------


def stage_a_index_sanity(
    index_path: Path, ds, model_config: ModernBertMiceConfig
) -> Dict[str, int]:
    logger.info("=" * 80)
    logger.info("STAGE A: PLAID Index Metadata Sanity Check")
    logger.info("=" * 80)

    metadata_file = index_path / "metadata.json"
    ext2int_file = index_path / "docid_ext2int.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_file}")
    with metadata_file.open("r") as fh:
        metadata = json.load(fh)

    with ext2int_file.open("r") as fh:
        ext2int = json.load(fh)

    corpus_size = ds.documents.documentcount or len(ext2int)

    logger.info(f"Index metadata: {metadata}")
    logger.info(f"ext2int entries: {len(ext2int)}")
    logger.info(f"Real corpus size (datamaestro): {corpus_size}")

    ok = True
    if metadata.get("num_docs") != len(ext2int):
        logger.error(
            f"[FAILURE] metadata.json num_docs ({metadata.get('num_docs')}) != "
            f"len(ext2int) ({len(ext2int)}). Index side-car files are inconsistent."
        )
        ok = False

    if len(ext2int) != corpus_size:
        logger.error(
            f"[FAILURE] Index covers {len(ext2int)} documents but the real corpus "
            f"has {corpus_size}. The index is stale or was built off a different/"
            f"filtered corpus (hypothesis 1: index built badly)."
        )
        ok = False
    else:
        logger.info("[OK] Index document count matches the real corpus size.")

    indexed_dim = metadata.get("dim")
    model_dim = model_config.hidden_size
    if indexed_dim != model_dim:
        logger.error(
            f"[FAILURE] Index embedding dim ({indexed_dim}) != checkpoint "
            f"hidden_size ({model_dim}). The index was very likely built with a "
            f"DIFFERENT model/checkpoint than the one being evaluated now."
        )
        ok = False
    else:
        logger.info(f"[OK] Index dim matches checkpoint hidden_size ({model_dim}).")

    logger.info(f"Index n_bits: {metadata.get('n_bits')}")

    if ok:
        logger.info("[SUCCESS] Stage A: index metadata looks consistent.")
    else:
        logger.warning(
            "[WARN] Stage A found inconsistencies above -- these alone could "
            "explain the reported bug. Continuing with Stage B/C regardless."
        )

    return {"corpus_size": corpus_size, "n_bits": metadata.get("n_bits")}


# ---------------------------------------------------------------------------
# Shared: exact online bottom-layer document encoding
# ---------------------------------------------------------------------------


def encode_document_exact(
    model: ModernBertMiceForSequenceClassification,
    tokenizer,
    doc_text: str,
    max_doc_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Recomputes the bottom-layer token embeddings for one document online.

    Matches exactly what `document_token_embeddings()` / `PlaidIndexBuilder` feed
    into fast-plaid at indexing time: `encode_documents(doc_ids, doc_mask)`.
    Batch size 1, no padding, so no mask-based filtering is needed.
    """
    enc = tokenizer(
        doc_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_doc_length,
        return_tensors="pt",
    )
    doc_ids = enc["input_ids"].to(device)
    doc_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        d_hidden, _ = model.encode_documents(doc_ids=doc_ids, doc_mask=doc_mask)
    return d_hidden[0]  # [num_tokens, dim], no padding to strip (batch size 1)


# ---------------------------------------------------------------------------
# Stage B: per-document reconstruction fidelity
# ---------------------------------------------------------------------------


def stage_b_reconstruction_fidelity(
    model: ModernBertMiceForSequenceClassification,
    tokenizer,
    index: PlaidIndex,
    ds,
    num_docs: int,
    max_doc_length: int,
    device: torch.device,
    seed: int,
) -> Dict[str, List[Tuple[str, torch.Tensor, torch.Tensor]]]:
    logger.info("=" * 80)
    logger.info("STAGE B: Per-Document Reconstruction Fidelity (real docs, real index)")
    logger.info("=" * 80)

    torch.manual_seed(seed)
    all_docs = list(ds.documents.iter_documents())
    if len(all_docs) > num_docs:
        idx = torch.randperm(len(all_docs))[:num_docs].tolist()
        sample = [all_docs[i] for i in idx]
    else:
        sample = all_docs

    logger.info(f"Sampling {len(sample)} real documents for reconstruction check...")

    length_mismatches = []
    cos_sims = []
    rel_l2_errs = []
    norm_ratios = []
    exact_norms = []
    recon_norms = []
    matched_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []

    for doc in sample:
        doc_id = doc.get("id") if isinstance(doc, dict) else getattr(doc, "id", None)
        text = (
            doc.get("text_item").text
            if isinstance(doc, dict)
            else doc["text_item"].text
        )
        if doc_id is None or not text:
            continue

        exact = encode_document_exact(model, tokenizer, text, max_doc_length, device)

        try:
            recon = index.get_document_tokens(doc_id).to(exact.device, dtype=exact.dtype)
        except KeyError:
            logger.warning(f"Doc id {doc_id!r} not found in PLAID index ext2int map!")
            continue

        if exact.shape[0] != recon.shape[0]:
            length_mismatches.append((doc_id, exact.shape[0], recon.shape[0]))
            continue

        exact_f = exact.float()
        recon_f = recon.float()
        cos = torch.nn.functional.cosine_similarity(exact_f, recon_f, dim=-1)
        rel_err = (exact_f - recon_f).norm(dim=-1) / exact_f.norm(dim=-1).clamp_min(1e-8)
        norm_ratio = recon_f.norm(dim=-1) / exact_f.norm(dim=-1).clamp_min(1e-8)

        cos_sims.append(cos)
        rel_l2_errs.append(rel_err)
        norm_ratios.append(norm_ratio)
        exact_norms.append(exact_f.norm(dim=-1))
        recon_norms.append(recon_f.norm(dim=-1))
        matched_pairs.append((doc_id, exact, recon))

    if length_mismatches:
        logger.error(
            f"[FAILURE] {len(length_mismatches)}/{len(sample)} documents have a "
            f"token-count mismatch between exact and reconstructed embeddings "
            f"(exact_len, recon_len shown): {length_mismatches[:5]}"
            f"{' ...' if len(length_mismatches) > 5 else ''}"
        )
        logger.error(
            "This points to an indexing/tokenization-config mismatch (hypothesis 1: "
            "index built with different tokenizer settings, e.g. max_doc_length), "
            "separate from PQ reconstruction fidelity."
        )

    if not cos_sims:
        logger.error(
            "[FAILURE] No documents had matching token counts -- cannot compute "
            "reconstruction fidelity. Fix the length mismatch above first."
        )
        return {"matched_pairs": []}

    all_cos = torch.cat(cos_sims)
    all_rel_err = torch.cat(rel_l2_errs)
    all_norm_ratio = torch.cat(norm_ratios)
    all_exact_norm = torch.cat(exact_norms)
    all_recon_norm = torch.cat(recon_norms)

    def _pct(t: torch.Tensor, q: float) -> float:
        return torch.quantile(t, q).item()

    logger.info(
        f"Compared {len(matched_pairs)}/{len(sample)} documents "
        f"({all_cos.numel()} total tokens)."
    )
    logger.info(
        f"Cosine similarity  -> mean={all_cos.mean().item():.4f}, "
        f"median={_pct(all_cos, 0.5):.4f}, p10={_pct(all_cos, 0.1):.4f}, "
        f"p90={_pct(all_cos, 0.9):.4f}, min={all_cos.min().item():.4f}"
    )
    logger.info(
        f"Relative L2 error  -> mean={all_rel_err.mean().item():.4f}, "
        f"median={_pct(all_rel_err, 0.5):.4f}, p10={_pct(all_rel_err, 0.1):.4f}, "
        f"p90={_pct(all_rel_err, 0.9):.4f}, max={all_rel_err.max().item():.4f}"
    )
    logger.info(
        f"Norm ratio (recon/exact) -> mean={all_norm_ratio.mean().item():.4f}, "
        f"median={_pct(all_norm_ratio, 0.5):.4f}, p10={_pct(all_norm_ratio, 0.1):.4f}, "
        f"p90={_pct(all_norm_ratio, 0.9):.4f}"
    )
    logger.info(
        f"Mean vector norm -> exact={all_exact_norm.mean().item():.4f} "
        f"(std={all_exact_norm.std().item():.4f}), "
        f"recon={all_recon_norm.mean().item():.4f} "
        f"(std={all_recon_norm.std().item():.4f})"
    )

    worst_idx = torch.argsort(all_cos)[:5].tolist()
    logger.info(f"5 worst-case token cosine similarities: {[round(all_cos[i].item(), 4) for i in worst_idx]}")

    if all_cos.mean().item() > 0.99:
        logger.info(
            "[SUCCESS] Stage B: reconstruction is near-exact (mean cosine > 0.99). "
            "PQ compression is unlikely to be the culprit -- check Stage C/D."
        )
    elif all_cos.mean().item() > 0.9:
        logger.warning(
            "[WARN] Stage B: reconstruction has noticeable but moderate error "
            "(mean cosine in (0.9, 0.99]). Could still meaningfully perturb "
            "MICE's top layers -- see Stage C for the actual score impact."
        )
    else:
        logger.error(
            "[FAILURE] Stage B: reconstruction is substantially lossy (mean cosine "
            "<= 0.9). This alone plausibly explains the reported score gap "
            "(hypothesis 1/3: index built with too-aggressive compression settings, "
            "or PQ is fundamentally too lossy for these un-normalized hidden states)."
        )

    return {"matched_pairs": matched_pairs}


# ---------------------------------------------------------------------------
# Stage C: downstream score sensitivity
# ---------------------------------------------------------------------------


def stage_c_score_sensitivity(
    model: ModernBertMiceForSequenceClassification,
    tokenizer,
    index: PlaidIndex,
    ds,
    num_queries: int,
    max_query_length: int,
    max_doc_length: int,
    device: torch.device,
) -> None:
    logger.info("=" * 80)
    logger.info("STAGE C: Downstream Score Sensitivity (online vs exact-bypass vs PLAID)")
    logger.info("=" * 80)

    topic_map = {t["id"]: t["text_item"].text for t in ds.topics.iter()}

    pairs: List[Tuple[str, str, str]] = []  # (query_text, doc_id, doc_text)
    for qrel in ds.assessments.iter():
        if qrel.topic_id not in topic_map:
            continue
        q_text = topic_map[qrel.topic_id]
        for assessment in qrel.assessments:
            doc = ds.documents.document_ext(assessment.doc_id)
            d_text = doc["text_item"].text
            pairs.append((q_text, assessment.doc_id, d_text))
            if len(pairs) >= num_queries:
                break
        if len(pairs) >= num_queries:
            break

    if not pairs:
        logger.error("[FAILURE] Stage C: no (query, relevant-doc) pairs found from qrels.")
        return

    logger.info(f"Evaluating {len(pairs)} real (query, relevant-doc) pairs...")

    online_scores, bypass_scores, plaid_scores = [], [], []
    rows = []

    for q_text, doc_id, d_text in pairs:
        q_enc = tokenizer(
            q_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        )
        d_enc = tokenizer(
            d_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_doc_length,
            return_tensors="pt",
        )
        query_ids = q_enc["input_ids"].to(device)
        query_mask = q_enc["attention_mask"].to(device)
        doc_ids = d_enc["input_ids"].to(device)
        doc_mask = d_enc["attention_mask"].to(device)

        with torch.no_grad():
            online_out = model(
                query_ids=query_ids,
                query_mask=query_mask,
                doc_ids=doc_ids,
                doc_mask=doc_mask,
            )
            online_score = online_out.logits.squeeze().item()

            exact_d_hidden, _ = model.encode_documents(
                doc_ids=doc_ids, doc_mask=doc_mask
            )
            bypass_out = model(
                query_ids=query_ids,
                query_mask=query_mask,
                doc_hidden=exact_d_hidden,
                doc_mask=doc_mask,
            )
            bypass_score = bypass_out.logits.squeeze().item()

            try:
                recon = index.get_document_tokens(doc_id).to(
                    device, dtype=exact_d_hidden.dtype
                )
            except KeyError:
                logger.warning(f"Doc id {doc_id!r} not found in PLAID index; skipping.")
                continue
            recon_padded = recon.unsqueeze(0)
            recon_mask = torch.ones(
                (1, recon.shape[0]), dtype=torch.long, device=device
            )
            plaid_out = model(
                query_ids=query_ids,
                query_mask=query_mask,
                doc_hidden=recon_padded,
                doc_mask=recon_mask,
            )
            plaid_score = plaid_out.logits.squeeze().item()

        online_scores.append(online_score)
        bypass_scores.append(bypass_score)
        plaid_scores.append(plaid_score)
        rows.append((doc_id, online_score, bypass_score, plaid_score))

    logger.info("-" * 80)
    logger.info(f"{'doc_id':<20}{'online':>12}{'exact_bypass':>16}{'plaid':>12}{'|on-by|':>10}{'|by-pl|':>10}")
    for doc_id, on, by, pl in rows:
        logger.info(
            f"{str(doc_id):<20}{on:>12.4f}{by:>16.4f}{pl:>12.4f}"
            f"{abs(on - by):>10.4f}{abs(by - pl):>10.4f}"
        )
    logger.info("-" * 80)

    on_t = torch.tensor(online_scores)
    by_t = torch.tensor(bypass_scores)
    pl_t = torch.tensor(plaid_scores)

    online_vs_bypass = (on_t - by_t).abs().mean().item()
    bypass_vs_plaid = (by_t - pl_t).abs().mean().item()
    online_vs_plaid = (on_t - pl_t).abs().mean().item()

    logger.info(
        f"Mean |online - exact_bypass| = {online_vs_bypass:.4f}  "
        f"(isolates the doc_hidden bypass code path from any compression)"
    )
    logger.info(
        f"Mean |exact_bypass - plaid|  = {bypass_vs_plaid:.4f}  "
        f"(isolates PLAID compression fidelity)"
    )
    logger.info(f"Mean |online - plaid|        = {online_vs_plaid:.4f}  (total reported gap)")

    logger.info("-" * 80)
    logger.info("VERDICT:")
    bypass_clean = online_vs_bypass < 0.05
    plaid_clean = bypass_vs_plaid < 0.05
    if bypass_clean and plaid_clean:
        logger.info(
            "  Online ~= exact_bypass ~= PLAID. Could not reproduce the reported "
            "score gap at this level. The bug likely lives in the retrieval-"
            "orchestration layer (MicePlaidRetrieverv2 / doc-id plumbing from the "
            "first-stage SPLADE retriever) rather than in encode->compress->"
            "reconstruct->score. See Stage D."
        )
    elif bypass_clean and not plaid_clean:
        logger.info(
            "  Online ~= exact_bypass, but exact_bypass diverges sharply from "
            "PLAID. CONFIRMED: the bug is in PLAID compression fidelity, "
            "consistent with Stage B's embedding-level error. Root cause is "
            "hypothesis (1) index built with too-aggressive settings, or (3) PQ "
            "is fundamentally too lossy for these un-normalized hidden states."
        )
    else:
        logger.info(
            "  Online ALREADY diverges from exact_bypass (before any compression "
            "is involved). The bug is in the forward() doc_hidden bypass code "
            "path itself, independent of PLAID/fast-plaid. PLAID vs exact_bypass "
            "divergence is secondary."
        )


# ---------------------------------------------------------------------------
# Stage D: doc-id coverage check
# ---------------------------------------------------------------------------


def stage_d_docid_coverage(index_path: Path, ds) -> None:
    logger.info("=" * 80)
    logger.info("STAGE D: Document-ID Coverage Check (index ext2int vs real corpus)")
    logger.info("=" * 80)

    with (index_path / "docid_ext2int.json").open("r") as fh:
        ext2int = json.load(fh)

    corpus_ids = set()
    for doc in ds.documents.iter_documents():
        doc_id = doc.get("id") if isinstance(doc, dict) else getattr(doc, "id", None)
        if doc_id is not None:
            corpus_ids.add(str(doc_id))

    indexed_ids = set(ext2int.keys())

    missing_from_index = corpus_ids - indexed_ids
    extra_in_index = indexed_ids - corpus_ids

    if missing_from_index:
        logger.error(
            f"[FAILURE] {len(missing_from_index)} corpus documents are missing "
            f"from the index's ext2int map (sample: "
            f"{list(missing_from_index)[:5]}). Those documents can never be "
            f"reranked correctly via PLAID."
        )
    if extra_in_index:
        logger.warning(
            f"[WARN] {len(extra_in_index)} index entries don't correspond to any "
            f"document in the current corpus (sample: {list(extra_in_index)[:5]}). "
            f"The index may be stale (built from an older/different corpus "
            f"snapshot) -- consistent with a Stage A hypothesis-1 finding."
        )

    if not missing_from_index and not extra_in_index:
        logger.info(
            f"[SUCCESS] Stage D: all {len(corpus_ids)} corpus document IDs are "
            f"covered 1:1 by the index's ext2int map."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose MICE v2 PLAID document-embedding reconstruction fidelity."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Trained MICE v2 checkpoint path.")
    parser.add_argument("--plaid-index-path", type=str, required=True, help="Path to an already-built PLAID index directory.")
    parser.add_argument("--dataset", type=str, default="org.beir.nfcorpus.test", help="Datamaestro dataset id.")
    parser.add_argument("--num-docs", type=int, default=50, help="Number of real documents to sample for Stage B.")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of (query, relevant-doc) pairs for Stage C.")
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-doc-length", type=int, default=1024, help="Must match what the PLAID index was built with.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", type=str, default="A,B,C,D", help="Comma-separated subset of stages to run.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    index_path = Path(args.plaid_index_path)
    stages = {s.strip().upper() for s in args.stages.split(",")}

    model = load_model(args.model_path, device, dtype, args.attn_implementation)
    tokenizer = load_tokenizer_for_model(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", model.config.pad_token_id)

    dataset_config, ds = load_dataset(args.dataset)
    index = PlaidIndex.C(
        documents=dataset_config.documents, index_path=index_path, compress_only=True
    ).instance()

    if "A" in stages:
        stage_a_index_sanity(index_path, ds, model.config)

    if "B" in stages:
        stage_b_reconstruction_fidelity(
            model, tokenizer, index, ds, args.num_docs, args.max_doc_length, device, args.seed
        )

    if "C" in stages:
        stage_c_score_sensitivity(
            model, tokenizer, index, ds, args.num_queries,
            args.max_query_length, args.max_doc_length, device,
        )

    if "D" in stages:
        stage_d_docid_coverage(index_path, ds)


if __name__ == "__main__":
    main()
