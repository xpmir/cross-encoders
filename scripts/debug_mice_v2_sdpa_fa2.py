"""
Layer-by-layer diagnostic script to pinpoint the exact layer where SDPA and FA2 diverge in MICE v2.
Usage:
    uv run python scripts/debug_mice_v2_sdpa_fa2.py --device cuda --dtype bfloat16
"""

import argparse
import logging
import torch
from transformers import AutoConfig, AutoTokenizer

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("debug_mice_v2_sdpa_fa2")


def main():
    parser = argparse.ArgumentParser(description="MICE v2 Layer-by-Layer SDPA vs FA2 Diagnostic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    logger.info(f"Running layer-by-layer diagnostic on device={device}, dtype={dtype}, model_path={args.model_path}")

    if args.model_path:
        base_config = ModernBertMiceConfig.from_pretrained(
            args.model_path,
        )
    else:
        base_config = ModernBertMiceConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=8,
            n_contextualization_layers=4,
            n_interaction_layers=4,
            pad_token_id=50283,
            sep_token_id=50283,
            cls_token_id=50281,
            local_attention=128,
            global_attn_every_n_layers=3,
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

    model_sdpa.eval()
    model_fa2.eval()

    # Build test batch with 512 tokens
    torch.manual_seed(42)
    seq0_valid = torch.cat([
        torch.tensor([50281]),
        torch.randint(100, 30000, (60,)),
        torch.tensor([50283]),
        torch.randint(100, 30000, (437,)),
        torch.tensor([50283])
    ]) # 500 tokens valid
    pad_id = base_config.pad_token_id
    seq0 = torch.cat([seq0_valid, torch.full((512 - len(seq0_valid),), pad_id, dtype=torch.long)])

    input_ids = seq0.unsqueeze(0).to(device) # Shape [1, 512]
    attention_mask = torch.zeros((1, 512), dtype=torch.long, device=device)
    attention_mask[0, :len(seq0_valid)] = 1

    logger.info("=" * 80)
    logger.info("STEP-BY-STEP LAYER BREAKDOWN")
    logger.info("=" * 80)

    with torch.no_grad():
        sdpa_outputs = model_sdpa.model(input_ids=input_ids, attention_mask=attention_mask)
        sdpa_bottom_hidden = sdpa_outputs.last_hidden_state

        fa2_outputs = model_fa2.model(input_ids=input_ids, attention_mask=attention_mask)
        fa2_bottom_hidden = fa2_outputs.last_hidden_state

        diff_bottom = (sdpa_bottom_hidden - fa2_bottom_hidden).abs().max().item()
        logger.info(f"1. Bottom Layers Output Max Abs Diff: {diff_bottom:.8f}")

        valid_mask = attention_mask[0].bool()
        valid_ids = input_ids[0][valid_mask]
        sep_indices = (valid_ids == 50283).nonzero(as_tuple=True)[0]
        sep_idx = sep_indices[0].item()

        q_seq_len = sep_idx + 1
        d_seq_len = len(valid_ids) - q_seq_len

        logger.info(f"ST Sequence Split: q_len={q_seq_len}, d_len={d_seq_len}")

        q_hidden_sdpa = sdpa_bottom_hidden[:, :q_seq_len, :]
        d_hidden_sdpa = sdpa_bottom_hidden[:, q_seq_len : q_seq_len + d_seq_len, :]

        q_hidden_fa2 = fa2_bottom_hidden[:, :q_seq_len, :]
        d_hidden_fa2 = fa2_bottom_hidden[:, q_seq_len : q_seq_len + d_seq_len, :]

        logger.info(f"Query Bottom Hidden Diff: {(q_hidden_sdpa - q_hidden_fa2).abs().max().item():.8f}")
        logger.info(f"Doc Bottom Hidden Diff:   {(d_hidden_sdpa - d_hidden_fa2).abs().max().item():.8f}")

        q_mask = torch.ones((1, q_seq_len), dtype=torch.long, device=device)
        d_mask = torch.ones((1, d_seq_len), dtype=torch.long, device=device)

        q_hidden_s = q_hidden_sdpa.clone()
        d_hidden_s = d_hidden_sdpa.clone()

        q_hidden_f = q_hidden_fa2.clone()
        d_hidden_f = d_hidden_fa2.clone()

        position_ids = torch.arange(q_seq_len, device=device).unsqueeze(0)
        q_pos_emb = model_sdpa.model.rotary_emb(q_hidden_s, position_ids, layer_type="full_attention")

        from MICE.modeling.mice_v2.modernbert_hf import create_sliding_window_mask
        from transformers.masking_utils import create_bidirectional_mask

        q_attn_mask = create_bidirectional_mask(model_sdpa.config, q_hidden_s, q_mask)
        q_sw_attn_mask = create_sliding_window_mask(q_mask, local_attention=model_sdpa.config.local_attention)

        for i in range(len(model_sdpa.top_layers)):
            top_layer_s = model_sdpa.top_layers[i]
            top_layer_f = model_fa2.top_layers[i]

            cross_s = top_layer_s.cross_attn(top_layer_s.cross_attn_norm(q_hidden_s), d_hidden_s, q_mask=q_mask, d_mask=d_mask)
            cross_f = top_layer_f.cross_attn(top_layer_f.cross_attn_norm(q_hidden_f), d_hidden_f, q_mask=q_mask, d_mask=d_mask)
            diff_cross = (cross_s - cross_f).abs().max().item()

            q_hidden_s = q_hidden_s + cross_s
            q_hidden_f = q_hidden_f + cross_f

            target_mask = q_sw_attn_mask if (q_sw_attn_mask is not None and top_layer_s.self_attn.sliding_window is not None) else q_attn_mask
            self_s, _ = top_layer_s.self_attn(top_layer_s.self_attn_norm(q_hidden_s), position_embeddings=q_pos_emb, attention_mask=target_mask)
            self_f, _ = top_layer_f.self_attn(top_layer_f.self_attn_norm(q_hidden_f), position_embeddings=q_pos_emb, attention_mask=None)
            diff_self = (self_s - self_f).abs().max().item()

            q_hidden_s = q_hidden_s + self_s
            q_hidden_f = q_hidden_f + self_f

            is_last = (i == len(model_sdpa.top_layers) - 1)
            if is_last:
                q_hidden_s = q_hidden_s[:, 0:1, :]
                q_hidden_f = q_hidden_f[:, 0:1, :]

            mlp_s = top_layer_s.mlp(top_layer_s.mlp_norm(q_hidden_s))
            mlp_f = top_layer_f.mlp(top_layer_f.mlp_norm(q_hidden_f))
            diff_mlp = (mlp_s - mlp_f).abs().max().item()

            q_hidden_s = q_hidden_s + mlp_s
            q_hidden_f = q_hidden_f + mlp_f

            logger.info(f"Top Layer {i} Diff -> Cross: {diff_cross:.8f} | Self: {diff_self:.8f} | MLP: {diff_mlp:.8f} | Cumulative Q: {(q_hidden_s - q_hidden_f).abs().max().item():.8f}")

        logit_s = model_sdpa.classifier(q_hidden_s[:, 0]).item()
        logit_f = model_fa2.classifier(q_hidden_f[:, 0]).item()
        logger.info(f"Final Logits -> SDPA: {logit_s:.8f} | FA2: {logit_f:.8f} | Diff: {abs(logit_s - logit_f):.8f}")


if __name__ == "__main__":
    main()
