#!/usr/bin/env python3
"""
Benchmark saved cross-encoder / MICE models.
Supports loading from local paths or remote paths (HOST:/lustre/...) using scp.
Currently computes theoretical FLOPs (standard and with document precomputation).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure src is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_utils import setup_logging
from flops import (
    get_transformer_params,
    compute_flops_model,
    get_hf_id,
    compute_flops_CrossEncoder,
)
from xpm_torch.huggingface import TorchHFHub
from transformers import AutoConfig

logger = logging.getLogger("benchmark_model")


def is_remote_path(path_str: str) -> bool:
    """
    Check if path is in remote scp format, e.g. 'HOST:/path' or 'user@HOST:/path'.
    Avoids false positives for Windows drive letters (e.g. 'C:/path').
    """
    if ":" in path_str:
        prefix, suffix = path_str.split(":", 1)
        if len(prefix) > 1 and suffix.startswith("/"):
            return True
    return False


def normalize_model_path(path_str: str, default_host: str = "JZ2") -> str:
    """
    Normalize model path. If path starts with '/lustre/' and local /lustre does not exist,
    automatically default to remote cluster host (default: JZ2).
    """
    if path_str.startswith("/lustre/") and not Path("/lustre").exists():
        logger.info(
            "Local '/lustre' does not exist; defaulting remote host to '%s' for '%s'",
            default_host,
            path_str,
        )
        return f"{default_host}:{path_str}"
    return path_str


def parse_remote_path(path_str: str) -> Tuple[str, str]:
    """Parse 'HOST:/path' or 'user@HOST:/path' into (host, remote_path)."""
    if not is_remote_path(path_str):
        raise ValueError(f"Path '{path_str}' is not a valid remote path specifier ('HOST:/path').")
    host, remote_path = path_str.split(":", 1)
    return host, remote_path


def fetch_remote_model_via_scp(
    remote_spec: str,
    target_dir: Optional[Path] = None,
    include_weights: bool = False,
    force_download: bool = False,
    port: Optional[int] = None,
) -> Path:
    """
    Fetch a remote model directory or definition file via scp.

    Args:
        remote_spec: Remote specifier of form 'HOST:/path' or 'user@HOST:/path'.
        target_dir: Local destination directory. Defaults to ~/.cache/cross_encoders/models/<model_name>.
        include_weights: If True, copies all model weights (*.safetensors). If False,
            only copies configuration files required for FLOPs benchmarks.
        force_download: If True, re-downloads even if cached files exist.
        port: SSH/SCP port if non-standard.

    Returns:
        Path to local directory containing model configuration.
    """
    host, remote_path = parse_remote_path(remote_spec)
    remote_path_clean = remote_path.rstrip("/")
    model_name = Path(remote_path_clean).name

    if target_dir is None:
        target_dir = Path.home() / ".cache" / "cross_encoders" / "models" / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    experimaestro_json = target_dir / "experimaestro.json"
    config_json = target_dir / "config.json"

    # Check if already present and valid
    if not force_download and (experimaestro_json.exists() or config_json.exists()):
        if not include_weights or list(target_dir.glob("*.safetensors")):
            logger.info("Using cached model in '%s' (use --force-download to refresh)", target_dir)
            return target_dir

    logger.info("Fetching model from '%s' to '%s' via scp...", remote_spec, target_dir)

    base_scp_cmd = ["scp"]
    if port is not None:
        base_scp_cmd.extend(["-P", str(port)])

    if include_weights:
        # Full directory copy
        cmd = base_scp_cmd + ["-r", f"{host}:{remote_path_clean}/.", str(target_dir)]
        logger.info("Executing full directory copy: %s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("scp command failed (exit code %d):\n%s", res.returncode, res.stderr)
            raise RuntimeError(f"scp failed with exit code {res.returncode}: {res.stderr.strip()}")
    else:
        # For FLOPs, we only need the definition file(s). Try experimaestro.json first.
        scp_success = False
        logger.info("Attempting to fetch 'experimaestro.json' from remote...")
        cmd_exp = base_scp_cmd + [f"{host}:{remote_path_clean}/experimaestro.json", str(experimaestro_json)]
        res_exp = subprocess.run(cmd_exp, capture_output=True, text=True)
        if res_exp.returncode == 0 and experimaestro_json.exists():
            scp_success = True
            logger.info("Successfully fetched 'experimaestro.json'.")
        else:
            logger.debug("Could not fetch experimaestro.json: %s", res_exp.stderr.strip())

        # Also attempt config.json if needed
        cmd_cfg = base_scp_cmd + [f"{host}:{remote_path_clean}/config.json", str(config_json)]
        res_cfg = subprocess.run(cmd_cfg, capture_output=True, text=True)
        if res_cfg.returncode == 0 and config_json.exists():
            scp_success = True
            logger.info("Successfully fetched 'config.json'.")

        # Also attempt results.csv if present
        results_csv = target_dir / "results.csv"
        if not results_csv.exists() or force_download:
            cmd_res = base_scp_cmd + [f"{host}:{remote_path_clean}/results.csv", str(results_csv)]
            res_r = subprocess.run(cmd_res, capture_output=True, text=True)
            if res_r.returncode == 0 and results_csv.exists():
                logger.info("Successfully fetched 'results.csv'.")

        # Also attempt modernbert_hf.py if present (needed by some custom model loaders)
        mb_file = target_dir / "modernbert_hf.py"
        if not mb_file.exists() or force_download:
            cmd_mb = base_scp_cmd + [f"{host}:{remote_path_clean}/modernbert_hf.py", str(mb_file)]
            subprocess.run(cmd_mb, capture_output=True, text=True)

        # Also attempt README.md if present (contains parameter count metadata)
        readme_file = target_dir / "README.md"
        if not readme_file.exists() or force_download:
            cmd_readme = base_scp_cmd + [f"{host}:{remote_path_clean}/README.md", str(readme_file)]
            subprocess.run(cmd_readme, capture_output=True, text=True)

        # If neither worked, maybe remote_path is a direct file or needs directory scp
        if not scp_success:
            if remote_path.endswith(".json"):
                dest_file = target_dir / Path(remote_path).name
                cmd_single = base_scp_cmd + [f"{host}:{remote_path}", str(dest_file)]
                logger.info("Remote path points directly to file. Fetching: %s", " ".join(cmd_single))
                res_single = subprocess.run(cmd_single, capture_output=True, text=True)
                if res_single.returncode == 0:
                    scp_success = True
            if not scp_success:
                logger.warning(
                    "Targeted scp for metadata files failed. Falling back to directory scp..."
                )
                cmd_dir = base_scp_cmd + ["-r", f"{host}:{remote_path_clean}/.", str(target_dir)]
                res_dir = subprocess.run(cmd_dir, capture_output=True, text=True)
                if res_dir.returncode != 0:
                    logger.error("Fallback scp directory copy failed:\n%s", res_dir.stderr)
                    raise RuntimeError(
                        f"Failed to scp remote model from '{remote_spec}': {res_dir.stderr.strip()}"
                    )
                scp_success = True

    return target_dir


def load_model_config(model_path: Path):
    """
    Loads model configuration from a local directory or file.
    Prefers experimaestro.json via TorchHFHub.pretrained_loader, falls back to config.json.
    """
    if model_path.is_file():
        model_path = model_path.parent

    experimaestro_path = model_path / "experimaestro.json"
    config_path = model_path / "config.json"

    if experimaestro_path.exists():
        logger.info("Loading Experimaestro model config from %s", experimaestro_path)
        loader_cfg = TorchHFHub.pretrained_loader(model_path, as_instance=False)
        return loader_cfg.model, "experimaestro"
    elif config_path.exists():
        logger.info("Loading HuggingFace AutoConfig from %s", config_path)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return hf_config, "huggingface"
    else:
        raise FileNotFoundError(
            f"No 'experimaestro.json' or 'config.json' found in model directory '{model_path}'."
        )


def run_flops_benchmark(
    model_path_or_spec: str,
    seq_len: int = 512,
    query_len: int = 32,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    include_weights: bool = False,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes the FLOPs benchmark for a given model (local or remote).
    """
    model_path_or_spec = normalize_model_path(model_path_or_spec)
    if is_remote_path(model_path_or_spec):
        local_dir = fetch_remote_model_via_scp(
            model_path_or_spec,
            target_dir=cache_dir,
            include_weights=include_weights,
            force_download=force_download,
            port=port,
        )
    else:
        local_dir = Path(model_path_or_spec).resolve()
        if not local_dir.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_dir}")

    config_obj, config_type = load_model_config(local_dir)

    if config_type == "experimaestro":
        model_class = type(config_obj).__name__
        hf_id = get_hf_id(config_obj)
        d, d_ff, nlayers = get_transformer_params(hf_id)

        model_flops, baseline_flops = compute_flops_model(
            config_obj, seq_len=seq_len, query_len=query_len, precompute_docs=False
        )
        model_flops_precomp, _ = compute_flops_model(
            config_obj, seq_len=seq_len, query_len=query_len, precompute_docs=True
        )

        n_ctx = getattr(config_obj, "n_contextualization_layers", None)
        n_inter = getattr(config_obj, "n_interaction_layers", None)
        if n_inter is None and n_ctx is not None:
            n_inter = nlayers - n_ctx

    elif config_type == "huggingface":
        model_class = "HFCrossEncoder"
        hf_id = getattr(config_obj, "_name_or_path", local_dir.name)
        d = config_obj.hidden_size
        d_ff = config_obj.intermediate_size
        nlayers = config_obj.num_hidden_layers
        model_flops, baseline_flops = compute_flops_CrossEncoder(
            d, d_ff, nlayers, seq_len=seq_len
        )
        model_flops_precomp = model_flops
        n_ctx = 0
        n_inter = nlayers
    else:
        raise ValueError(f"Unknown config_type: {config_type}")

    speedup = baseline_flops / model_flops if model_flops > 0 else float("inf")
    speedup_precomp = (
        baseline_flops / model_flops_precomp if model_flops_precomp > 0 else float("inf")
    )

    eval_results = {}
    results_csv_file = local_dir / "results.csv"
    if results_csv_file.exists():
        try:
            import pandas as pd
            df_eval = pd.read_csv(results_csv_file)
            eval_results = df_eval.set_index("dataset").to_dict(orient="index")
            logger.info("Loaded evaluation metrics for %d datasets from %s", len(eval_results), results_csv_file)
        except Exception as e:
            logger.warning("Could not parse results.csv: %s", e)

    n_params = None
    readme_path = local_dir / "README.md"
    if readme_path.exists():
        try:
            content = readme_path.read_text()
            match = re.search(r"Total Parameters:\s*([^\n\(\)]+)\s*\(([^M\)]+M)\)", content, re.IGNORECASE)
            if match:
                n_params = match.group(2).strip()
            else:
                match2 = re.search(r"Total Parameters:\s*([^\n\(\)]+)", content, re.IGNORECASE)
                if match2:
                    n_params = match2.group(1).strip()
        except Exception as e:
            logger.debug("Could not parse README.md for n_params: %s", e)

    if n_params is None:
        st_path = local_dir / "model.safetensors"
        if st_path.is_file():
            try:
                from safetensors import safe_open
                total_p = 0
                with safe_open(str(st_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        total_p += f.get_tensor(key).numel()
                n_params = f"{total_p / 1e6:.2f}M"
            except Exception as e:
                logger.debug("Could not count parameters from safetensors: %s", e)

    results = {
        "model_path": model_path_or_spec,
        "local_path": str(local_dir),
        "model_class": model_class,
        "hf_id": hf_id,
        "config_type": config_type,
        "n_params": n_params,
        "d": d,
        "d_ff": d_ff,
        "nlayers": nlayers,
        "n_contextualization_layers": n_ctx,
        "n_interaction_layers": n_inter,
        "seq_len": seq_len,
        "query_len": query_len,
        "doc_len": seq_len - query_len - 3,
        "baseline_Gflops": baseline_flops / 1e9,
        "standard": {
            "model_Gflops": model_flops / 1e9,
            "pct_of_baseline": (model_flops / baseline_flops) * 100,
            "speedup": speedup,
        },
        "precompute_docs": {
            "model_Gflops": model_flops_precomp / 1e9,
            "pct_of_baseline": (model_flops_precomp / baseline_flops) * 100,
            "speedup": speedup_precomp,
        },
        "evaluation": eval_results,
    }
    return results


def print_results(res: Dict[str, Any]):
    """Format and print the benchmark results cleanly."""
    line = "=" * 80
    print("\n" + line)
    print(f"BENCHMARK RESULTS: FLOPs Analysis")
    print(line)
    print(f"Model Source     : {res['model_path']}")
    print(f"Model Class      : {res['model_class']} ({res['config_type']})")
    print(f"Backbone HF ID   : {res['hf_id']}")
    if res.get("n_params"):
        print(f"Parameters       : {res['n_params']}")
    print(f"Hidden Dim (d)   : {res['d']}")
    print(f"MLP Dim (d_ff)   : {res['d_ff']}")
    print(f"Total Layers (L) : {res['nlayers']}")
    if res["n_contextualization_layers"] is not None:
        print(f"Contextualization: {res['n_contextualization_layers']} layers")
    if res["n_interaction_layers"] is not None:
        print(f"Interaction      : {res['n_interaction_layers']} layers")
    print(f"Sequence Length  : {res['seq_len']} (query: {res['query_len']}, doc: {res['doc_len']})")
    print("-" * 80)
    print(f"Baseline Full Cross-Encoder : {res['baseline_Gflops']:.2f} GFLOPs (100.0%)")
    print("-" * 80)
    print(
        f"Online (No Precompute)      : {res['standard']['model_Gflops']:.2f} GFLOPs "
        f"({res['standard']['pct_of_baseline']:.1f}% baseline, {res['standard']['speedup']:.2f}x speedup)"
    )
    print(
        f"Cached Docs (Precomputed)   : {res['precompute_docs']['model_Gflops']:.2f} GFLOPs "
        f"({res['precompute_docs']['pct_of_baseline']:.1f}% baseline, {res['precompute_docs']['speedup']:.2f}x speedup)"
    )
    if res.get("evaluation"):
        print("-" * 80)
        print("Information Retrieval Evaluation Metrics:")
        print("-" * 80)
        eval_dict = res["evaluation"]
        summary_keys = ["msmarco_dev", "In Domain", "BEIR13", "BEIR13 Decontaminated", "mean"]
        for k in summary_keys:
            if k in eval_dict:
                metrics_str = ", ".join(f"{m}: {v:.2f}" for m, v in eval_dict[k].items() if v is not None)
                print(f"  {k:<25}: {metrics_str}")
    print(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cross-Encoder / MICE model FLOPs from local or remote (scp) paths."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model: local path or remote 'HOST:/lustre/...'",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Total sequence length (default: 512)",
    )
    parser.add_argument(
        "--query-len",
        type=int,
        default=32,
        help="Query length (default: 32)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Custom local directory to store files downloaded via scp",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force scp download even if cached files exist",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Download model weights (*.safetensors) via scp (default: False for FLOPs)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="SSH/SCP port if non-standard",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON string",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results as CSV line (n_params,Flops,Flops_precomp,Baseline_flops,frac_baseline,frac_precompute)",
    )
    parser.add_argument(
        "--csv-gflops",
        action="store_true",
        help="Output results as CSV line with FLOPs in GFLOPs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON results file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )

    args = parser.parse_args()

    # If csv output requested, default logging to WARNING so table stdout is clean
    default_log_level = logging.WARNING if (args.csv or args.csv_gflops or args.json) else logging.INFO
    setup_logging(level=logging.DEBUG if args.verbose else default_log_level)
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else default_log_level)

    results = run_flops_benchmark(
        model_path_or_spec=args.model_path,
        seq_len=args.seq_len,
        query_len=args.query_len,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        include_weights=args.include_weights,
        port=args.port,
    )

    if args.csv:
        print("n_params,Flops,Flops_precomp,Baseline_flops,frac_baseline,frac_precompute")
        n_p = results.get("n_params") or "N/A"
        f = int(results["standard"]["model_Gflops"] * 1e9)
        f_p = int(results["precompute_docs"]["model_Gflops"] * 1e9)
        b = int(results["baseline_Gflops"] * 1e9)
        fb = results["standard"]["pct_of_baseline"] / 100
        fp = results["precompute_docs"]["pct_of_baseline"] / 100
        print(f"{n_p},{f},{f_p},{b},{fb:.4f},{fp:.4f}")
    elif args.csv_gflops:
        print("n_params,Flops,Flops_precomp,Baseline_flops,frac_baseline,frac_precompute")
        n_p = results.get("n_params") or "N/A"
        f = f"{results['standard']['model_Gflops']:.2f}"
        f_p = f"{results['precompute_docs']['model_Gflops']:.2f}"
        b = f"{results['baseline_Gflops']:.2f}"
        fb = results["standard"]["pct_of_baseline"] / 100
        fp = results["precompute_docs"]["pct_of_baseline"] / 100
        print(f"{n_p},{f},{f_p},{b},{fb:.4f},{fp:.4f}")
    elif args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
