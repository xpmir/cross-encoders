#!/usr/bin/env python3
"""
Benchmark script comparing MICE v2 Unified Forward pass against Legacy Two-Pass execution.

Models evaluated:
  1. Remote Baseline: VictorMorand/Mice-P-130m with trust_remote_code=True
  2. Local Unified:   ModernBertMiceForSequenceClassification (unified_bottom_pass=True)
  3. Local Legacy:    ModernBertMiceForSequenceClassification (unified_bottom_pass=False)

Usage:
  uv run python scripts/benchmark_mice_v2_unified.py [--device mps|cuda|cpu] [--batch-sizes 8,16,32,64] [--doc-lens 128,256,512]
"""

import argparse
import logging
import math
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_mice_v2_unified")


def sync_device(device: torch.device):
    """Synchronizes target hardware backend to ensure accurate latency measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def create_synthetic_batch(
    batch_size: int,
    query_len: int,
    doc_len: int,
    vocab_size: int,
    cls_token_id: int,
    sep_token_id: int,
    pad_token_id: int,
    device: torch.device,
    skew_padding: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates synthetic query and document batches with realistic padding distribution."""
    query_ids = torch.full((batch_size, query_len), pad_token_id, dtype=torch.long, device=device)
    query_mask = torch.zeros((batch_size, query_len), dtype=torch.long, device=device)

    doc_ids = torch.full((batch_size, doc_len), pad_token_id, dtype=torch.long, device=device)
    doc_mask = torch.zeros((batch_size, doc_len), dtype=torch.long, device=device)

    for b in range(batch_size):
        # Queries: uniform length between 12 and query_len
        q_act_len = torch.randint(max(10, query_len // 2), query_len + 1, (1,)).item() if skew_padding else query_len
        # Docs: variable length between doc_len // 2 and doc_len
        d_act_len = torch.randint(max(20, doc_len // 2), doc_len + 1, (1,)).item() if skew_padding else doc_len

        # Populate query tokens
        query_ids[b, 0] = cls_token_id
        if q_act_len > 2:
            query_ids[b, 1 : q_act_len - 1] = torch.randint(100, vocab_size - 100, (q_act_len - 2,), device=device)
        query_ids[b, q_act_len - 1] = sep_token_id
        query_mask[b, :q_act_len] = 1

        # Populate doc tokens
        doc_ids[b, 0] = cls_token_id
        if d_act_len > 2:
            doc_ids[b, 1 : d_act_len - 1] = torch.randint(100, vocab_size - 100, (d_act_len - 2,), device=device)
        doc_ids[b, d_act_len - 1] = sep_token_id
        doc_mask[b, :d_act_len] = 1

    return query_ids, query_mask, doc_ids, doc_mask


def measure_forward(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    num_iters: int,
) -> Tuple[float, float, float]:
    """Measures inference forward latency (mean_ms, median_ms, std_ms)."""
    q_ids, q_mask, d_ids, d_mask = batch
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask)
            sync_device(device)

    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            sync_device(device)
            t0 = time.perf_counter()
            _ = model(query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask)
            sync_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    median_ms = latencies[len(latencies) // 2]
    var_ms = sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)
    std_ms = math.sqrt(var_ms)

    return mean_ms, median_ms, std_ms


def measure_train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    num_iters: int,
) -> Tuple[float, float, float]:
    """Measures forward + backward latency (mean_ms, median_ms, std_ms)."""
    q_ids, q_mask, d_ids, d_mask = batch
    model_dtype = next(model.parameters()).dtype
    labels = torch.zeros((q_ids.shape[0],), dtype=model_dtype, device=device)
    model.train()

    # Warmup
    for _ in range(warmup_iters):
        out = model(query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        sync_device(device)

    latencies = []
    for _ in range(num_iters):
        sync_device(device)
        t0 = time.perf_counter()
        out = model(query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        sync_device(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    median_ms = latencies[len(latencies) // 2]
    var_ms = sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)
    std_ms = math.sqrt(var_ms)

    return mean_ms, median_ms, std_ms


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark MICE v2 Unified vs Legacy forward pass.")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--batch-sizes", type=str, default="8,16,32,64")
    parser.add_argument("--doc-lens", type=str, default="128,256,512")
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--model-id", type=str, default="VictorMorand/Mice-P-130m")
    parser.add_argument("--include-train", action="store_true", help="Also benchmark training step (fwd + bwd)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve dtype
    if args.dtype == "auto":
        target_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        target_dtype = getattr(torch, args.dtype)

    # Resolve attn implementation
    if args.attn_implementation == "auto":
        target_attn_impl = "flash_attention_2" if device.type == "cuda" else "sdpa"
    else:
        target_attn_impl = args.attn_implementation

    logger.info(f"Target benchmark device: {device} | dtype: {target_dtype} | attn_impl: {target_attn_impl}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_cap = torch.cuda.get_device_capability(device)
        logger.info(f"CUDA Device: {gpu_name} (Compute Capability: {gpu_cap[0]}.{gpu_cap[1]})")

    logger.info(f"Loading models from HF Hub: {args.model_id}...")

    # Load remote baseline
    logger.info(f"Loading Remote Baseline (trust_remote_code=True, attn_impl={target_attn_impl})...")
    model_remote = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        attn_implementation=target_attn_impl,
        torch_dtype=target_dtype,
    ).to(device)

    # Load local implementation
    logger.info(f"Loading Local Implementation (attn_impl={target_attn_impl})...")
    model_unified = ModernBertMiceForSequenceClassification.from_pretrained(
        args.model_id,
        attn_implementation=target_attn_impl,
        torch_dtype=target_dtype,
    ).to(device)
    model_unified.config.unified_bottom_pass = True

    model_legacy = ModernBertMiceForSequenceClassification.from_pretrained(
        args.model_id,
        attn_implementation=target_attn_impl,
        torch_dtype=target_dtype,
    ).to(device)
    model_legacy.config.unified_bottom_pass = False

    config = model_unified.config
    vocab_size = config.vocab_size
    cls_token_id = getattr(config, "cls_token_id", 50281)
    sep_token_id = getattr(config, "sep_token_id", 50282)
    pad_token_id = getattr(config, "pad_token_id", 50283)

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    doc_lens = [int(d.strip()) for d in args.doc_lens.split(",") if d.strip()]

    results: List[Dict] = []

    logger.info("Starting benchmarking across batch sizes and document lengths...")
    for bsz in batch_sizes:
        for dlen in doc_lens:
            logger.info(f"\n--- Testing Batch Size = {bsz}, Query Len = {args.query_len}, Doc Len = {dlen} ---")
            batch = create_synthetic_batch(
                batch_size=bsz,
                query_len=args.query_len,
                doc_len=dlen,
                vocab_size=vocab_size,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                pad_token_id=pad_token_id,
                device=device,
                skew_padding=True,
            )

            # Numerical parity check
            model_remote.eval()
            model_unified.eval()
            with torch.no_grad():
                out_remote = model_remote(query_ids=batch[0], query_mask=batch[1], doc_ids=batch[2], doc_mask=batch[3])
                out_unified = model_unified(query_ids=batch[0], query_mask=batch[1], doc_ids=batch[2], doc_mask=batch[3])
                max_diff = (out_remote.logits - out_unified.logits).abs().max().item()

            if max_diff > 1e-4:
                logger.error(f"Numerical disparity detected! Max diff = {max_diff:.6f}")
                raise RuntimeError(f"Numerical parity check failed for B={bsz}, D={dlen}")
            else:
                logger.info(f"Parity verified: max absolute difference = {max_diff:.2e}")

            # Measure Inference Forward
            logger.info("Timing Remote Baseline (Inference)...")
            rem_mean, rem_med, rem_std = measure_forward(
                model_remote, batch, device, args.warmup_iters, args.num_iters
            )

            logger.info("Timing Local Legacy (unified_bottom_pass=False)...")
            leg_mean, leg_med, leg_std = measure_forward(
                model_legacy, batch, device, args.warmup_iters, args.num_iters
            )

            logger.info("Timing Local Unified (unified_bottom_pass=True)...")
            uni_mean, uni_med, uni_std = measure_forward(
                model_unified, batch, device, args.warmup_iters, args.num_iters
            )

            speedup_vs_remote = rem_mean / uni_mean if uni_mean > 0 else 0.0
            speedup_vs_legacy = leg_mean / uni_mean if uni_mean > 0 else 0.0
            throughput_pairs_sec = (bsz / (uni_mean / 1000.0))

            logger.info(
                f"Result: Remote={rem_mean:.2f}ms | Legacy={leg_mean:.2f}ms | Unified={uni_mean:.2f}ms | "
                f"Speedup vs Remote={speedup_vs_remote:.2f}x | Speedup vs Legacy={speedup_vs_legacy:.2f}x | "
                f"Throughput={throughput_pairs_sec:.1f} pairs/s"
            )

            res_entry = {
                "batch_size": bsz,
                "doc_len": dlen,
                "query_len": args.query_len,
                "mode": "eval",
                "remote_ms": rem_mean,
                "legacy_ms": leg_mean,
                "unified_ms": uni_mean,
                "speedup_remote": speedup_vs_remote,
                "speedup_legacy": speedup_vs_legacy,
                "throughput": throughput_pairs_sec,
            }
            results.append(res_entry)

            # Optional Training Step Benchmark
            if args.include_train:
                logger.info("Timing Remote Baseline (Train Fwd + Bwd)...")
                rem_tr_mean, _, _ = measure_train_step(
                    model_remote, batch, device, args.warmup_iters, args.num_iters
                )
                logger.info("Timing Local Legacy (Train Fwd + Bwd)...")
                leg_tr_mean, _, _ = measure_train_step(
                    model_legacy, batch, device, args.warmup_iters, args.num_iters
                )
                logger.info("Timing Local Unified (Train Fwd + Bwd)...")
                uni_tr_mean, _, _ = measure_train_step(
                    model_unified, batch, device, args.warmup_iters, args.num_iters
                )
                tr_speedup_rem = rem_tr_mean / uni_tr_mean if uni_tr_mean > 0 else 0.0
                tr_speedup_leg = leg_tr_mean / uni_tr_mean if uni_tr_mean > 0 else 0.0
                logger.info(
                    f"Train Step: Remote={rem_tr_mean:.2f}ms | Legacy={leg_tr_mean:.2f}ms | Unified={uni_tr_mean:.2f}ms | "
                    f"Speedup vs Remote={tr_speedup_rem:.2f}x | Speedup vs Legacy={tr_speedup_leg:.2f}x"
                )
                results.append({
                    "batch_size": bsz,
                    "doc_len": dlen,
                    "query_len": args.query_len,
                    "mode": "train",
                    "remote_ms": rem_tr_mean,
                    "legacy_ms": leg_tr_mean,
                    "unified_ms": uni_tr_mean,
                    "speedup_remote": tr_speedup_rem,
                    "speedup_legacy": tr_speedup_leg,
                    "throughput": (bsz / (uni_tr_mean / 1000.0)),
                })

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Device: {device} | Query Len: {args.query_len} | Warmup: {args.warmup_iters} | Iters: {args.num_iters}\n")

    eval_results = [r for r in results if r["mode"] == "eval"]
    print("### Inference (Forward Pass)")
    print("| Batch Size | Doc Length | Baseline (Remote) | Legacy (Local) | Unified (Local) | Speedup vs Baseline | Speedup vs Legacy | Throughput (pairs/s) |")
    print("|:----------:|:----------:|:-----------------:|:--------------:|:---------------:|:-------------------:|:-----------------:|:--------------------:|")
    for r in eval_results:
        print(
            f"| {r['batch_size']:<10} | {r['doc_len']:<10} | "
            f"{r['remote_ms']:>9.2f} ms   | {r['legacy_ms']:>9.2f} ms  | {r['unified_ms']:>10.2f} ms  | "
            f"**{r['speedup_remote']:>5.2f}x**           | "
            f"**{r['speedup_legacy']:>5.2f}x**          | "
            f"{r['throughput']:>12.1f}         |"
        )

    if args.include_train:
        train_results = [r for r in results if r["mode"] == "train"]
        print("\n### Training (Forward + Backward Pass)")
        print("| Batch Size | Doc Length | Baseline (Remote) | Legacy (Local) | Unified (Local) | Speedup vs Baseline | Speedup vs Legacy | Throughput (pairs/s) |")
        print("|:----------:|:----------:|:-----------------:|:--------------:|:---------------:|:-------------------:|:-----------------:|:--------------------:|")
        for r in train_results:
            print(
                f"| {r['batch_size']:<10} | {r['doc_len']:<10} | "
                f"{r['remote_ms']:>9.2f} ms   | {r['legacy_ms']:>9.2f} ms  | {r['unified_ms']:>10.2f} ms  | "
                f"**{r['speedup_remote']:>5.2f}x**           | "
                f"**{r['speedup_legacy']:>5.2f}x**          | "
                f"{r['throughput']:>12.1f}         |"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark()
