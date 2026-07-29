#!/usr/bin/env python3
"""
Verification and Parity Benchmark Script for MICE v2 Flash Attention 2 (FA2) Unpadding Optimization.

Verifies:
  1. Output logits numerical parity between SDPA (padded 3D) and FA2 (unpadded 1D) paths.
  2. Backward pass gradient flow and gradient parity.
  3. Latency & memory throughput benchmarks for SDPA vs FA2 across uniform and skewed padding batches.

Usage:
  uv run python scripts/test_mice_v2_fa2_parity.py [--device cuda] [--dtype bfloat16] [--batch-size 32] [--max-doc-len 512] [--warmup-iters 15] [--num-iters 50]
"""

import argparse
import logging
import time
import torch
import torch.nn as nn

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
    is_fa2_available,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mice_v2_fa2_parity")


def check_flash_attn_package() -> bool:
    """Checks if native tri-dao flash_attn C++/CUDA extension is installed."""
    try:
        import flash_attn
        logger.info("[ENV CHECK] Native `flash_attn` (tri-dao C++/CUDA extension) is INSTALLED!")
        return True
    except ImportError:
        logger.warning(
            "[ENV CHECK] `flash_attn` (tri-dao C++/CUDA package) is NOT installed in this Python environment.\n"
            "   HuggingFace is falling back to `kernels-community/flash-attn2` (Triton/PyTorch fallback kernel).\n"
            "   For maximum CUDA speedup, install native flash-attn via:\n"
            "   --> uv pip install flash-attn --no-build-isolation\n"
        )
        return False


def build_synthetic_batch(
    batch_size: int,
    device: torch.device,
    max_q_len: int = 64,
    max_d_len: int = 512,
    highly_skewed: bool = False,
):
    """Builds variable-length synthetic query and document batches with random padding."""
    torch.manual_seed(42)
    q_ids_list, d_ids_list = [], []
    q_mask_list, d_mask_list = [], []

    for i in range(batch_size):
        if highly_skewed and (i % 5 != 0):
            # 80% of batch items are short (e.g. len 16-32 for query, 64-128 for doc)
            q_len = torch.randint(16, 32, (1,)).item()
            d_len = torch.randint(64, 128, (1,)).item()
        else:
            # 20% are long / max length (up to max_q_len and max_d_len)
            q_len = torch.randint(max_q_len // 2, max_q_len + 1, (1,)).item()
            d_len = torch.randint(max_d_len // 2, max_d_len + 1, (1,)).item()

        q_seq = torch.randint(100, 30000, (q_len,), device=device)
        d_seq = torch.randint(100, 30000, (d_len,), device=device)

        q_ids_list.append(q_seq)
        d_ids_list.append(d_seq)
        q_mask_list.append(torch.ones(q_len, dtype=torch.long, device=device))
        d_mask_list.append(torch.ones(d_len, dtype=torch.long, device=device))

    query_ids = torch.nn.utils.rnn.pad_sequence(q_ids_list, batch_first=True, padding_value=0)
    query_mask = torch.nn.utils.rnn.pad_sequence(q_mask_list, batch_first=True, padding_value=0)
    doc_ids = torch.nn.utils.rnn.pad_sequence(d_ids_list, batch_first=True, padding_value=0)
    doc_mask = torch.nn.utils.rnn.pad_sequence(d_mask_list, batch_first=True, padding_value=0)

    labels = torch.rand(batch_size, 1, device=device)

    return query_ids, query_mask, doc_ids, doc_mask, labels


def test_parity_and_gradients(device: torch.device, dtype: torch.dtype):
    logger.info("=" * 70)
    logger.info("STAGE 1: Forward & Backward Parity Check (SDPA vs FA2 Unpadded)")
    logger.info("=" * 70)

    base_config = ModernBertMiceConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=8,
        n_contextualization_layers=4,
        n_interaction_layers=4,
    )

    config_sdpa = ModernBertMiceConfig(**base_config.to_dict())
    config_sdpa._attn_implementation = "sdpa"

    config_fa2 = ModernBertMiceConfig(**base_config.to_dict())
    config_fa2._attn_implementation = "flash_attention_2"

    logger.info("Initializing SDPA and FA2 models with identical weights...")
    torch.manual_seed(1337)
    model_sdpa = ModernBertMiceForSequenceClassification(config_sdpa).to(device=device, dtype=dtype)

    torch.manual_seed(1337)
    model_fa2 = ModernBertMiceForSequenceClassification(config_fa2).to(device=device, dtype=dtype)

    model_fa2.load_state_dict(model_sdpa.state_dict())

    query_ids, query_mask, doc_ids, doc_mask, labels = build_synthetic_batch(
        batch_size=8, device=device, max_q_len=64, max_d_len=256
    )
    labels = labels.to(dtype=dtype)

    # Forward pass SDPA
    model_sdpa.eval()
    with torch.no_grad():
        out_sdpa = model_sdpa(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    # Forward pass FA2
    model_fa2.eval()
    with torch.no_grad():
        out_fa2 = model_fa2(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    logits_sdpa = out_sdpa.logits
    logits_fa2 = out_fa2.logits

    diff = (logits_sdpa - logits_fa2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info(f"Forward Logits SDPA sample: {logits_sdpa[:3].squeeze(-1).tolist()}")
    logger.info(f"Forward Logits FA2  sample: {logits_fa2[:3].squeeze(-1).tolist()}")
    logger.info(f"Logits Abs Difference -> Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

    # Set tolerance matching floating-point format precision (bfloat16 has 7 mantissa bits, eps ~0.0078)
    atol = 0.02 if dtype == torch.bfloat16 else (1e-2 if dtype == torch.float16 else 1e-4)
    rtol = 0.02 if dtype == torch.bfloat16 else (1e-2 if dtype == torch.float16 else 1e-4)
    is_close = torch.allclose(logits_sdpa, logits_fa2, atol=atol, rtol=rtol)

    if is_close:
        logger.info(f"[SUCCESS] Forward pass logits parity check PASSED (atol={atol}, rtol={rtol})!")
    else:
        logger.warning(f"[WARNING] Logits difference exceeded tolerance (atol={atol}, rtol={rtol}).")

    # Backward Pass & Gradient Parity
    logger.info("Testing Backward pass & gradient flow...")
    model_sdpa.train()
    model_fa2.train()

    model_sdpa.zero_grad()
    model_fa2.zero_grad()

    out_sdpa_tr = model_sdpa(
        query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels
    )
    out_fa2_tr = model_fa2(
        query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels
    )

    out_sdpa_tr.loss.backward()
    out_fa2_tr.loss.backward()

    grad_sdpa = model_sdpa.classifier.weight.grad
    grad_fa2 = model_fa2.classifier.weight.grad

    if grad_sdpa is not None and grad_fa2 is not None:
        grad_diff = (grad_sdpa - grad_fa2).abs().max().item()
        logger.info(f"Classifier Weight Grad Max Abs Difference: {grad_diff:.6f}")
        if torch.allclose(grad_sdpa, grad_fa2, atol=atol, rtol=rtol):
            logger.info(f"[SUCCESS] Backward pass gradient parity check PASSED (atol={atol}, rtol={rtol})!")
        else:
            logger.warning("[WARNING] Gradient difference exceeded tolerance.")
    else:
        logger.error("[ERROR] Gradients were None after backward pass!")


def benchmark_throughput(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    max_q_len: int = 64,
    max_d_len: int = 512,
    warmup_iters: int = 15,
    num_iters: int = 50,
    highly_skewed: bool = False,
):
    label_str = "High-Padding Skewed" if highly_skewed else "Standard"
    logger.info("=" * 70)
    logger.info(f"STAGE 2: Latency & Throughput Benchmark [{label_str} Batches]")
    logger.info("=" * 70)

    config_sdpa = ModernBertMiceConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=22,
        n_contextualization_layers=18,
        n_interaction_layers=4,
        _attn_implementation="sdpa",
    )

    config_fa2 = ModernBertMiceConfig(**config_sdpa.to_dict())
    config_fa2._attn_implementation = "flash_attention_2"

    logger.info(f"Building full 22-layer ModernBERT MICE models (B={batch_size}, max_q={max_q_len}, max_d={max_d_len})...")
    model_sdpa = ModernBertMiceForSequenceClassification(config_sdpa).to(device=device, dtype=dtype)
    model_fa2 = ModernBertMiceForSequenceClassification(config_fa2).to(device=device, dtype=dtype)

    model_sdpa.train()
    model_fa2.train()

    query_ids, query_mask, doc_ids, doc_mask, labels = build_synthetic_batch(
        batch_size=batch_size,
        device=device,
        max_q_len=max_q_len,
        max_d_len=max_d_len,
        highly_skewed=highly_skewed,
    )
    labels = labels.to(dtype=dtype)

    total_q_tokens = query_mask.sum().item()
    total_d_tokens = doc_mask.sum().item()
    total_tokens_per_batch = total_q_tokens + total_d_tokens
    max_tokens_padded = (query_ids.shape[1] + doc_ids.shape[1]) * query_ids.shape[0]
    padding_pct = (1.0 - total_tokens_per_batch / max_tokens_padded) * 100.0

    logger.info(
        f"Batch config: B={batch_size}, Real Tokens={total_tokens_per_batch}, Padded Tokens={max_tokens_padded} "
        f"({padding_pct:.1f}% Padding)"
    )

    # 1. Dedicated Warmup SDPA
    logger.info(f"Running {warmup_iters} CUDA warmup iterations for SDPA...")
    for _ in range(warmup_iters):
        out = model_sdpa(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels)
        out.loss.backward()
        model_sdpa.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark SDPA
    logger.info(f"Running {num_iters} timed benchmark iterations for SDPA...")
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = model_sdpa(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels)
        out.loss.backward()
        model_sdpa.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    sdpa_time = (t1 - t0) / num_iters * 1000.0
    sdpa_tps = (total_tokens_per_batch * num_iters) / (t1 - t0)

    logger.info(f"SDPA Mode: {sdpa_time:.2f} ms/iter | Throughput: {sdpa_tps:.1f} tokens/sec")

    # 2. Dedicated Warmup FA2
    logger.info(f"Running {warmup_iters} CUDA warmup iterations for FA2...")
    for _ in range(warmup_iters):
        out = model_fa2(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels)
        out.loss.backward()
        model_fa2.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark FA2
    logger.info(f"Running {num_iters} timed benchmark iterations for FA2...")
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = model_fa2(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask, labels=labels)
        out.loss.backward()
        model_fa2.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    fa2_time = (t1 - t0) / num_iters * 1000.0
    fa2_tps = (total_tokens_per_batch * num_iters) / (t1 - t0)

    speedup = (sdpa_time / fa2_time) if fa2_time > 0 else 0.0
    logger.info(f"FA2 Mode:  {fa2_time:.2f} ms/iter | Throughput: {fa2_tps:.1f} tokens/sec")
    logger.info(f"[BENCHMARK SUMMARY ({label_str})] FA2 Speedup vs SDPA: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Test MICE v2 FA2 Unpadding Parity and Speed.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (e.g. 'cuda' or 'cpu').")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Target data type.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for throughput testing.")
    parser.add_argument("--max-query-len", type=int, default=64, help="Max query token length.")
    parser.add_argument("--max-doc-len", type=int, default=512, help="Max document token length.")
    parser.add_argument("--warmup-iters", type=int, default=15, help="Number of warmup iterations.")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of timed benchmark iterations.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    logger.info(f"Starting MICE v2 FA2 Parity Test on device={device}, dtype={dtype}")

    if device.type != "cuda":
        logger.warning(
            "CUDA device is not active in this environment. FA2 unpadded path requires CUDA SM 8.0+. "
            "Running SDPA fallback verification locally."
        )
        test_parity_and_gradients(device=torch.device("cpu"), dtype=torch.float32)
        return

    fa2_avail = is_fa2_available()
    if not fa2_avail:
        logger.warning(
            "FlashAttention 2 is not available/supported in this Python environment. "
            "Install `flash-attn` or `kernels-community/flash-attn2` to enable FA2 benchmarking."
        )
        return

    check_flash_attn_package()
    test_parity_and_gradients(device=device, dtype=dtype)
    benchmark_throughput(
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_q_len=args.max_query_len,
        max_d_len=args.max_doc_len,
        warmup_iters=args.warmup_iters,
        num_iters=args.num_iters,
        highly_skewed=False,
    )
    benchmark_throughput(
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_q_len=args.max_query_len,
        max_d_len=args.max_doc_len,
        warmup_iters=args.warmup_iters,
        num_iters=args.num_iters,
        highly_skewed=True,
    )


if __name__ == "__main__":
    main()
