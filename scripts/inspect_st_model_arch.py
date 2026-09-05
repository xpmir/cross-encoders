#!/usr/bin/env python3
"""
Diagnostic script to load and inspect STCrossEncoder model architecture
using the exact Experimaestro loading pipeline (SimpleModuleLoader + STCrossEncoder).

Usage:
  uv run python scripts/inspect_st_model_arch.py --path /path/to/checkpoint/model
  uv run python scripts/inspect_st_model_arch.py --path cross-encoder/ms-marco-MiniLM-L6-v2
"""

import argparse
import logging
from pathlib import Path

from xpm_torch.module import SimpleModuleLoader
from xpmir.neural.sentence_transformers import STCrossEncoder, st_cross_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inspect_st_model_arch")


def print_st_architecture(scorer: STCrossEncoder, path: str):
    """Print the complete architecture and module tree of an STCrossEncoder."""
    st_model = scorer.st_model

    print("\n" + "=" * 70)
    print(f"       STCrossEncoder Model Architecture Inspection")
    print(f"       Path / ID: {path}")
    print("=" * 70)

    # 1. Top-level properties
    print(f"\n[1] STCrossEncoder Configuration:")
    print(f"  - Model ID:                  {scorer.model_id}")
    print(f"  - Max Length:                {scorer.max_length}")
    print(f"  - Pref Attention Impl:       {scorer.pref_attn_implementation}")
    print(f"  - Is Initialized:            {scorer._initialized}")
    print(f"  - Class Type:                {type(scorer).__name__}")

    # 2. SentenceTransformers Module Pipeline
    print(f"\n[2] SentenceTransformers Module Pipeline:")
    if hasattr(st_model, "_modules") and st_model._modules:
        for idx, (name, mod) in enumerate(st_model._modules.items()):
            param_count = sum(p.numel() for p in mod.parameters())
            trainable_count = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  Module [{idx}] key='{name}': {type(mod).__name__}")
            print(f"    - Parameters: {param_count:,} (Trainable: {trainable_count:,})")
            if hasattr(mod, "auto_model"):
                config = getattr(mod.auto_model, "config", None)
                attn_impl = getattr(config, "_attn_implementation", "N/A") if config else "N/A"
                archs = getattr(config, "architectures", "N/A") if config else "N/A"
                print(f"    - Base Model Arch: {archs}")
                print(f"    - Active Attention: '{attn_impl}'")
    else:
        print(f"  Native CrossEncoder (non-sequential module layout): {st_model}")

    # 3. Activation Function & Model Card Data
    print(f"\n[3] ST Model Settings:")
    print(f"  - Activation Function:       {getattr(st_model, 'activation_fn', 'None')}")
    model_card = getattr(st_model, "model_card_data", None)
    if model_card:
        print(f"  - Model Card Name:           {getattr(model_card, 'model_name', 'N/A')}")

    # 4. Total Parameter Summary
    total_params = sum(p.numel() for p in scorer.parameters())
    trainable_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    print(f"\n[4] Total Parameter Summary:")
    print(f"  - Total Parameters:          {total_params:,}")
    print(f"  - Trainable Parameters:      {trainable_params:,}")

    # 5. Detailed PyTorch Layer Tree
    print(f"\n[5] PyTorch Sub-module Hierarchy:")
    for name, module in st_model.named_children():
        print(f"  +-- {name}: {type(module).__name__}")
        for child_name, child_module in module.named_children():
            print(f"      +-- {child_name}: {type(child_module).__name__}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect loaded STCrossEncoder model architecture and modules."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to checkpoint directory (containing modules.json or model.safetensors) or HF model ID.",
    )
    parser.add_argument(
        "--base-id",
        "-b",
        type=str,
        default="jhu-clsp/ettin-encoder-68m",
        help="Base model ID to initialize the scorer spec (default: jhu-clsp/ettin-encoder-68m).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512).",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        help="Preferred attention implementation e.g. 'sdpa' or 'flash_attention_2' (default: 'sdpa').",
    )
    args = parser.parse_args()

    input_path = Path(args.path)

    logger.info("Initializing base STCrossEncoder with base_id='%s'", args.base_id)
    scorer_cfg = st_cross_scorer(
        model_id=args.base_id,
        max_length=args.max_length,
        pref_attn_implementation=args.attn_impl,
    )

    if input_path.exists():
        logger.info("Checkpoint path '%s' exists on disk.", input_path)
        logger.info("Executing SimpleModuleLoader on path...")
        # Experimaestro SimpleModuleLoader.C takes the Config object (scorer_cfg), NOT the instance
        loader_cfg = SimpleModuleLoader.C(value=scorer_cfg, path=input_path)
        loader_inst = loader_cfg.instance()
        loader_inst.execute()
        scorer_inst = loader_inst.model
    else:
        logger.info(
            "Path '%s' does not exist as local path. Treating as model ID and calling load_model directly...",
            args.path,
        )
        scorer_inst = scorer_cfg.instance()
        scorer_inst.initialize()
        scorer_inst.load_model(input_path)

    # Print architecture
    print_st_architecture(scorer_inst, args.path)


if __name__ == "__main__":
    main()
