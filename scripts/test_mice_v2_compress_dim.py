#!/usr/bin/env python3
"""
Verification Script for MICE v2 `compress_dim` (compressed top interaction layers).

Verifies:
  1. For compress_dim in {1.0, 2.0, 4.0} x each compress_init_option (skip the
     option axis when compress_dim==1.0): the model builds, runs a forward
     pass (SDPA, CPU-friendly), and produces finite logits of the expected
     shape.
  2. For "random_projection"/"svd_projection": loading real pretrained
     weights via `mice_v2_scorer(..., compress_dim=..., compress_init_option=...)`
     + `InitMICEV2ModernBERTFromHFID` runs without shape errors.
  3. SDPA-vs-FlashAttention-2 forward-pass parity with compress_dim=2.0 active,
     for each compress_init_option -- exercises the new `top_rotary_emb`/
     `rotary_for_top` code path under real FlashAttention-2 execution.
     Requires CUDA SM 8.0+ hardware with FlashAttention-2 available; this
     stage is SKIPPED (not failed) on incompatible hardware.

Usage:
  uv run python scripts/test_mice_v2_compress_dim.py [--device cuda] [--dtype bfloat16] [--hf-id jhu-clsp/ettin-encoder-17m]
"""

import argparse
import logging

import torch

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
    is_fa2_available,
)
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mice_v2_compress_dim")

COMPRESS_INIT_OPTIONS = ["random", "random_projection", "svd_projection"]


def check_flash_attn_hardware() -> bool:
    """Returns True if CUDA SM 8.0+ with FlashAttention-2 is available, else False (never raises)."""
    if not torch.cuda.is_available():
        return False
    if not is_fa2_available():
        return False
    return True


def build_synthetic_inputs(device: torch.device, pad_token_id: int):
    torch.manual_seed(42)
    query_ids = torch.randint(100, 30000, (2, 16), device=device)
    query_mask = torch.ones((2, 16), dtype=torch.long, device=device)
    doc_ids = torch.randint(100, 30000, (2, 32), device=device)
    doc_mask = torch.ones((2, 32), dtype=torch.long, device=device)
    return query_ids, query_mask, doc_ids, doc_mask


def stage1_forward_shapes(device: torch.device):
    logger.info("=" * 70)
    logger.info("STAGE 1: Forward pass shapes/finiteness across compress_dim x compress_init_option")
    logger.info("=" * 70)

    for compress_dim in (1.0, 2.0, 4.0):
        options = [None] if compress_dim == 1.0 else COMPRESS_INIT_OPTIONS
        for compress_init_option in options:
            kwargs = dict(
                hidden_size=64,
                num_attention_heads=8,
                num_hidden_layers=4,
                intermediate_size=128,
                n_contextualization_layers=2,
                pad_token_id=50283,
                compress_dim=compress_dim,
            )
            if compress_init_option is not None:
                kwargs["compress_init_option"] = compress_init_option

            config = ModernBertMiceConfig(**kwargs)
            config._attn_implementation = "sdpa"
            model = ModernBertMiceForSequenceClassification(config).to(device=device)
            model.eval()

            query_ids, query_mask, doc_ids, doc_mask = build_synthetic_inputs(
                device, config.pad_token_id
            )
            with torch.no_grad():
                out = model(
                    query_ids=query_ids,
                    query_mask=query_mask,
                    doc_ids=doc_ids,
                    doc_mask=doc_mask,
                )

            assert out.logits.shape == (2, 1), f"Unexpected logits shape {out.logits.shape}"
            assert torch.isfinite(out.logits).all(), "Non-finite logits"
            logger.info(
                f"[OK] compress_dim={compress_dim}, compress_init_option={compress_init_option}: "
                f"logits={out.logits.squeeze(-1).tolist()}"
            )

    logger.info("[SUCCESS] Stage 1 passed.")


def stage2_real_backbone_init(hf_id: str):
    logger.info("=" * 70)
    logger.info(f"STAGE 2: Real pretrained backbone init ({hf_id}) under compress_dim=2.0")
    logger.info("=" * 70)

    for compress_init_option in ("random_projection", "svd_projection"):
        scorer_cfg, init_tasks = mice_v2_scorer(
            hf_id=hf_id,
            n_contextualization_layers=2,
            compress_dim=2.0,
            compress_init_option=compress_init_option,
        )
        for task in init_tasks:
            inst_task = task.instance()
            inst_task.execute()
            scorer = inst_task.scorer

        scorer.eval()
        with torch.no_grad():
            query_ids, query_mask, doc_ids, doc_mask = build_synthetic_inputs(
                torch.device("cpu"), scorer.hf_model.config.pad_token_id
            )
            out = scorer.hf_model(
                query_ids=query_ids,
                query_mask=query_mask,
                doc_ids=doc_ids,
                doc_mask=doc_mask,
            )
        assert torch.isfinite(out.logits).all()
        logger.info(f"[OK] compress_init_option={compress_init_option}: logits finite, shape={out.logits.shape}")

    logger.info("[SUCCESS] Stage 2 passed.")


def stage3_fa2_parity(device: torch.device, dtype: torch.dtype):
    logger.info("=" * 70)
    logger.info("STAGE 3: SDPA vs FlashAttention-2 parity under compress_dim=2.0")
    logger.info("=" * 70)

    if not check_flash_attn_hardware():
        logger.warning("[SKIPPED] Stage 3 requires CUDA SM 8.0+ hardware with FlashAttention-2 available.")
        return

    for compress_init_option in COMPRESS_INIT_OPTIONS:
        base_config = ModernBertMiceConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=8,
            n_contextualization_layers=4,
            n_interaction_layers=4,
            pad_token_id=50283,
            sep_token_id=50283,
            cls_token_id=50281,
            compress_dim=2.0,
            compress_init_option=compress_init_option,
        )

        config_sdpa = ModernBertMiceConfig(**base_config.to_dict())
        config_sdpa._attn_implementation = "sdpa"

        config_fa2 = ModernBertMiceConfig(**base_config.to_dict())
        config_fa2._attn_implementation = "flash_attention_2"

        torch.manual_seed(1337)
        model_sdpa = ModernBertMiceForSequenceClassification(config_sdpa).to(device=device, dtype=dtype)

        torch.manual_seed(1337)
        model_fa2 = ModernBertMiceForSequenceClassification(config_fa2).to(device=device, dtype=dtype)
        model_fa2.load_state_dict(model_sdpa.state_dict())

        query_ids, query_mask, doc_ids, doc_mask = build_synthetic_inputs(device, base_config.pad_token_id)

        model_sdpa.eval()
        model_fa2.eval()
        with torch.no_grad():
            out_sdpa = model_sdpa(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)
            out_fa2 = model_fa2(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

        diff = (out_sdpa.logits - out_fa2.logits).abs()
        max_diff = diff.max().item()

        atol = 0.02 if dtype == torch.bfloat16 else (1e-2 if dtype == torch.float16 else 1e-4)
        rtol = atol
        is_close = torch.allclose(out_sdpa.logits, out_fa2.logits, atol=atol, rtol=rtol)

        logger.info(
            f"compress_init_option={compress_init_option}: max_diff={max_diff:.6f} "
            f"(atol={atol}) -> {'OK' if is_close else 'MISMATCH'}"
        )
        if not is_close:
            raise ValueError(
                f"[FAILURE] SDPA/FA2 parity failed for compress_init_option={compress_init_option} "
                f"(max_diff={max_diff:.6f}, atol={atol})."
            )

    logger.info("[SUCCESS] Stage 3 passed.")


def main():
    parser = argparse.ArgumentParser(description="Test MICE v2 compress_dim correctness and SDPA/FA2 parity.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for Stage 1/2 (Stage 3 always needs CUDA).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Target data type for Stage 3.",
    )
    parser.add_argument(
        "--hf-id",
        type=str,
        default="jhu-clsp/ettin-encoder-17m",
        help="HF checkpoint id used for Stage 2's real-backbone init check.",
    )
    parser.add_argument("--skip-stage2", action="store_true", help="Skip the real-backbone download/init stage.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    stage1_forward_shapes(device)
    if not args.skip_stage2:
        stage2_real_backbone_init(args.hf_id)
    stage3_fa2_parity(torch.device("cuda") if torch.cuda.is_available() else device, dtype)


if __name__ == "__main__":
    main()
