#!/usr/bin/env python3
"""Score calibration analysis between cross-encoder teacher scores and LateOn multi-vector scores.

Samples (query, document, teacher_score) pairs from cross-encoder/ettin-reranker-v1-data,
computes LateOn late-interaction scores (both MaxSim and MeanMaxSim), performs affine
regressions (student -> teacher and teacher -> student), and generates distribution
and calibration match plots.

Usage:
    uv run python scripts/analyze_score_calibration.py --num_samples 500
    uv run python scripts/analyze_score_calibration.py --num_samples 1000 --subsets rerank_msmarco rerank_hotpotqa
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import torch
from datasets import load_dataset
from sentence_transformers import MultiVectorEncoder
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("score_calibration")

DEFAULT_SUBSETS = [
    "rerank_msmarco",
    "rerank_hotpotqa",
    "rerank_nq",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and calibrate score distributions between ettin-reranker-v1 teacher and LateOn multi-vector encoder."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="lightonai/LateOn-unsupervised",
        help="HuggingFace model ID for LateOn / MultiVectorEncoder",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help="Subsets of cross-encoder/ettin-reranker-v1-data to sample from",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Total number of samples to evaluate across subsets",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model encoding",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="figures/score_distribution_lateon_vs_teacher.png",
        help="Path where the comparison plot will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'mps', 'cpu'). Defaults to auto-detection.",
    )
    return parser.parse_args()


def sample_dataset(
    subsets: List[str], target_samples: int
) -> List[Dict[str, Any]]:
    """Samples query-document-teacher tuples evenly across specified subsets."""
    samples_per_subset = max(1, target_samples // len(subsets))
    logger.info(
        f"Sampling ~{samples_per_subset} samples from each of {len(subsets)} subsets: {subsets}"
    )

    collected: List[Dict[str, Any]] = []
    for subset in subsets:
        logger.info(f"Streaming from subset: '{subset}' (streaming=True)...")
        try:
            ds = load_dataset(
                "cross-encoder/ettin-reranker-v1-data",
                subset,
                split="train",
                streaming=True,
            ).shuffle(buffer_size=10_000, seed=42)
            count = 0
            for item in ds:
                q = item.get("query")
                d = item.get("document")
                label = item.get("label")

                if q is None or d is None or label is None:
                    continue

                collected.append(
                    {
                        "subset": subset,
                        "query": q,
                        "document": d,
                        "teacher_score": float(label),
                    }
                )
                count += 1
                if count >= samples_per_subset:
                    break

            logger.info(f"Collected {count} samples from '{subset}'.")
        except Exception as e:
            logger.warning(f"Could not load subset '{subset}': {e}. Skipping.")

    logger.info(f"Total samples collected: {len(collected)}")
    if not collected:
        raise RuntimeError("No samples could be collected from the dataset.")
    return collected


def compute_lateon_scores(
    model: MultiVectorEncoder,
    samples: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes MaxSim and MeanMaxSim scores using the LateOn MultiVectorEncoder."""
    logger.info("Computing LateOn scores in batches...")

    all_maxsim: List[float] = []
    all_meanmaxsim: List[float] = []
    all_teachers: List[float] = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Encoding pairs"):
        batch = samples[i : i + batch_size]
        queries = [b["query"] for b in batch]
        documents = [b["document"] for b in batch]
        teachers = [b["teacher_score"] for b in batch]

        with torch.no_grad():
            q_emb = model.encode_query(
                queries, convert_to_numpy=False, normalize_embeddings=True
            )
            d_emb = model.encode_document(
                documents, convert_to_numpy=False, normalize_embeddings=True
            )

            # MaxSim: length_normalize=False
            scores_max = model.similarity_pairwise(
                q_emb, d_emb, length_normalize=False
            )
            # MeanMaxSim: length_normalize=True
            scores_mean = model.similarity_pairwise(
                q_emb, d_emb, length_normalize=True
            )

        all_maxsim.extend(scores_max.cpu().tolist())
        all_meanmaxsim.extend(scores_mean.cpu().tolist())
        all_teachers.extend(teachers)

    return (
        np.array(all_teachers, dtype=np.float64),
        np.array(all_maxsim, dtype=np.float64),
        np.array(all_meanmaxsim, dtype=np.float64),
    )


def fit_affine_regressions(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fits linear regressions in both directions (x -> y and y -> x) with correlation metrics."""
    # Student (x) -> Teacher (y): y = w * x + b
    res_xy = stats.linregress(x, y)
    w, b = res_xy.slope, res_xy.intercept
    r_val, p_val = res_xy.rvalue, res_xy.pvalue
    r2 = r_val**2

    # Teacher (y) -> Student (x): x = alpha * y + beta
    res_yx = stats.linregress(y, x)
    alpha, beta = res_yx.slope, res_yx.intercept

    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    # Predictions and RMSE
    y_pred = w * x + b
    rmse_before = np.sqrt(np.mean((y - x) ** 2))
    rmse_after = np.sqrt(np.mean((y - y_pred) ** 2))

    return {
        "slope_student_to_teacher": w,
        "bias_student_to_teacher": b,
        "slope_teacher_to_student": alpha,
        "bias_teacher_to_student": beta,
        "pearson_r": r_val,
        "spearman_rho": spearman_rho,
        "r2": r2,
        "rmse_raw": rmse_before,
        "rmse_calibrated": rmse_after,
        "y_pred": y_pred,
    }


def generalized_sigmoid(
    y: np.ndarray, s_min: float, s_max: float, y0: float, tau: float
) -> np.ndarray:
    """Generalized 4-parameter logistic/sigmoid function mapping teacher logits y -> student scores s.

    s(y) = s_min + (s_max - s_min) / (1 + exp(-(y - y0) / tau))
    """
    return s_min + (s_max - s_min) / (1.0 + np.exp(-np.clip((y - y0) / tau, -50.0, 50.0)))


def fit_sigmoid_regression(
    teacher: np.ndarray,
    student: np.ndarray,
) -> Dict[str, Any]:
    """Fits a 4-parameter generalized sigmoid mapping teacher logits -> student scores.

    Model: s(y) = s_min + (s_max - s_min) / (1 + exp(-(y - y0) / tau))
    """
    s_min_init = float(np.percentile(student, 5))
    s_max_init = float(np.percentile(student, 95))
    y0_init = float(np.median(teacher))
    tau_init = float(max(teacher.std() / 4.0, 1.0))

    p0 = [s_min_init, s_max_init, y0_init, tau_init]
    s_lo = min(-10.0, float(student.min() - 2 * student.std()))
    s_hi = max(50.0, float(student.max() + 2 * student.std()))
    bounds = (
        [s_lo, s_lo, -100.0, 0.01],
        [s_hi, s_hi, 100.0, 50.0],
    )

    try:
        popt, _ = curve_fit(
            generalized_sigmoid,
            teacher,
            student,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        s_min, s_max, y0, tau = popt
        s_pred = generalized_sigmoid(teacher, *popt)

        ss_tot = np.sum((student - np.mean(student)) ** 2)
        ss_res = np.sum((student - s_pred) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((student - s_pred) ** 2)))

        return {
            "success": True,
            "popt": popt,
            "s_min": s_min,
            "s_max": s_max,
            "y0": y0,
            "tau": tau,
            "r2": r2,
            "rmse": rmse,
            "s_pred": s_pred,
        }
    except Exception as e:
        logger.warning(f"Sigmoid curve fitting failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def print_stats_table(name: str, arr: np.ndarray):
    """Prints formatted summary statistics for a 1D array."""
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"| {name:<18} | {arr.mean():7.2f} | {arr.std():6.2f} | {arr.min():7.2f} | {p5:6.2f} | {p50:6.2f} | {p95:6.2f} | {arr.max():7.2f} |")


def plot_distributions(
    teacher: np.ndarray,
    maxsim: np.ndarray,
    meanmaxsim: np.ndarray,
    reg_max: Dict[str, Any],
    reg_mean: Dict[str, Any],
    sig_max: Dict[str, Any],
    sig_mean: Dict[str, Any],
    output_path: Path,
):
    """Generates a 2-row x 3-column figure visualizing distributions, regressions, and calibrations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # ROW 1: MaxSim (Unnormalized late-interaction sum)
    # -------------------------------------------------------------
    # 1A: Raw distributions comparison
    ax = axes[0, 0]
    ax.hist(teacher, bins=40, density=True, alpha=0.5, color="#1f77b4", label=f"Teacher Logits (μ={teacher.mean():.1f}, σ={teacher.std():.1f})")
    ax.hist(maxsim, bins=40, density=True, alpha=0.5, color="#ff7f0e", label=f"LateOn MaxSim (μ={maxsim.mean():.1f}, σ={maxsim.std():.1f})")
    ax.set_title("1A. Raw Distributions: Teacher vs LateOn MaxSim", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 1B: Scatter plot & linear / sigmoid regression
    ax = axes[0, 1]
    ax.scatter(maxsim, teacher, alpha=0.35, edgecolors="none", color="#2ca02c", s=25, label="Data pairs")
    line_x = np.linspace(maxsim.min(), maxsim.max(), 100)
    line_y = reg_max["slope_student_to_teacher"] * line_x + reg_max["bias_student_to_teacher"]
    w, b = reg_max["slope_student_to_teacher"], reg_max["bias_student_to_teacher"]
    ax.plot(line_x, line_y, color="red", lw=2, linestyle="--", label=f"Affine: y = {w:.3f}·s + {b:.2f}\nR² = {reg_max['r2']:.3f}")
    if sig_max.get("success"):
        grid_y = np.linspace(teacher.min(), teacher.max(), 300)
        grid_s = generalized_sigmoid(grid_y, *sig_max["popt"])
        ax.plot(
            grid_s,
            grid_y,
            color="#1f77b4",
            lw=2.5,
            label=f"Sigmoid: s(y)\nR² = {sig_max['r2']:.3f}, RMSE = {sig_max['rmse']:.2f}",
        )
    ax.set_title("1B. Fits: LateOn MaxSim ↔ Teacher Logits", fontsize=12, fontweight="bold")
    ax.set_xlabel("LateOn MaxSim Score (s)")
    ax.set_ylabel("Teacher Logit (y)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 1C: Transformed match
    ax = axes[0, 2]
    transformed_maxsim = reg_max["y_pred"]
    ax.hist(teacher, bins=40, density=True, alpha=0.5, color="#1f77b4", label="Teacher Logits (Target)")
    ax.hist(transformed_maxsim, bins=40, density=True, alpha=0.5, color="#d62728", label=f"Calibrated MaxSim (Affine)\nRMSE: {reg_max['rmse_calibrated']:.2f}")
    ax.set_title("1C. Distribution Match: Teacher vs Calibrated MaxSim", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # ROW 2: MeanMaxSim (Length-normalized token similarity)
    # -------------------------------------------------------------
    # 2A: Raw distributions comparison
    ax = axes[1, 0]
    ax.hist(teacher, bins=40, density=True, alpha=0.5, color="#1f77b4", label=f"Teacher Logits (μ={teacher.mean():.1f}, σ={teacher.std():.1f})")
    ax.hist(meanmaxsim, bins=40, density=True, alpha=0.5, color="#9467bd", label=f"LateOn MeanMaxSim (μ={meanmaxsim.mean():.2f}, σ={meanmaxsim.std():.2f})")
    ax.set_title("2A. Raw Distributions: Teacher vs LateOn MeanMaxSim", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2B: Scatter plot & regressions (Affine vs Sigmoid)
    ax = axes[1, 1]
    ax.scatter(meanmaxsim, teacher, alpha=0.35, edgecolors="none", color="#8c564b", s=25, label="Data pairs")
    line_x = np.linspace(meanmaxsim.min(), meanmaxsim.max(), 100)
    w_m, b_m = reg_mean["slope_student_to_teacher"], reg_mean["bias_student_to_teacher"]
    line_y = w_m * line_x + b_m
    ax.plot(line_x, line_y, color="red", lw=2, linestyle="--", label=f"Affine: y = {w_m:.2f}·s + {b_m:.2f}\nR² = {reg_mean['r2']:.3f}")
    if sig_mean.get("success"):
        grid_y = np.linspace(teacher.min(), teacher.max(), 300)
        grid_s = generalized_sigmoid(grid_y, *sig_mean["popt"])
        ax.plot(
            grid_s,
            grid_y,
            color="#1f77b4",
            lw=2.5,
            label=f"Sigmoid: s(y)\n[s_min={sig_mean['s_min']:.2f}, s_max={sig_mean['s_max']:.2f}]\n[y0={sig_mean['y0']:.1f}, τ={sig_mean['tau']:.2f}]\nR² = {sig_mean['r2']:.3f}, RMSE = {sig_mean['rmse']:.3f}",
        )
    ax.set_title("2B. Fits: LateOn MeanMaxSim ↔ Teacher Logits", fontsize=12, fontweight="bold")
    ax.set_xlabel("LateOn MeanMaxSim Score (s)")
    ax.set_ylabel("Teacher Logit (y)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2C: Transformed match & teacher-to-student calibration
    ax = axes[1, 2]
    alpha = reg_mean["slope_teacher_to_student"]
    beta = reg_mean["bias_teacher_to_student"]
    teacher_scaled_fitted = alpha * teacher + beta
    teacher_scaled_heuristic = 0.02 * teacher + 0.70
    rmse_aff = np.sqrt(np.mean((meanmaxsim - teacher_scaled_fitted) ** 2))
    rmse_rule = np.sqrt(np.mean((meanmaxsim - teacher_scaled_heuristic) ** 2))

    ax.hist(meanmaxsim, bins=40, density=True, alpha=0.4, color="#9467bd", label="LateOn MeanMaxSim (Target)")
    ax.hist(
        teacher_scaled_fitted,
        bins=40,
        density=True,
        histtype="step",
        lw=2,
        color="#2ca02c",
        label=f"Affine ({alpha:.3f}·y + {beta:.2f}, RMSE={rmse_aff:.3f})",
    )
    if sig_mean.get("success"):
        ax.hist(
            sig_mean["s_pred"],
            bins=40,
            density=True,
            alpha=0.5,
            color="#1f77b4",
            label=f"Sigmoid Calib (R²={sig_mean['r2']:.3f}, RMSE={sig_mean['rmse']:.3f})",
        )
    ax.hist(
        teacher_scaled_heuristic,
        bins=40,
        density=True,
        histtype="step",
        lw=1.5,
        linestyle="--",
        color="black",
        label=f"Default Rule (0.020·y + 0.70, RMSE={rmse_rule:.3f})",
    )
    ax.set_title("2C. Calibrated Teacher vs Target MeanMaxSim", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibrated Score value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        "Empirical Score Distribution & Calibration Analysis (Affine vs Sigmoid)\nTeacher (mxbai-rerank-large-v2) vs LateOn (MultiVectorEncoder)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved successfully to: {output_path}")


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: '{device}'")

    # 1. Sample dataset
    samples = sample_dataset(args.subsets, args.num_samples)

    # 2. Load model
    logger.info(f"Loading LateOn MultiVectorEncoder from '{args.model_id}'...")
    model = MultiVectorEncoder(args.model_id, device=device)

    # 3. Compute scores
    teachers, maxsim, meanmaxsim = compute_lateon_scores(
        model, samples, args.batch_size
    )

    # 4. Affine regressions
    reg_max = fit_affine_regressions(maxsim, teachers)
    reg_mean = fit_affine_regressions(meanmaxsim, teachers)

    # 5. Generalized Sigmoid regressions (Teacher -> Student)
    sig_max = fit_sigmoid_regression(teachers, maxsim)
    sig_mean = fit_sigmoid_regression(teachers, meanmaxsim)

    # 6. Print Summary Tables
    print("\n" + "=" * 90)
    print("SCORE DISTRIBUTION DESCRIPTIVE STATISTICS")
    print("=" * 90)
    print(f"| {'Metric / Score':<18} | {'Mean':>7} | {'Std':>6} | {'Min':>7} | {'P5':>6} | {'Median':>6} | {'P95':>6} | {'Max':>7} |")
    print("|" + "-" * 20 + "|" + "-" * 9 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")
    print_stats_table("Teacher Logits", teachers)
    print_stats_table("LateOn MaxSim", maxsim)
    print_stats_table("LateOn MeanMaxSim", meanmaxsim)
    print("=" * 90)

    print("\n" + "=" * 90)
    print("AFFINE REGRESSION RESULTS & OPTIMAL COEFFICIENTS")
    print("=" * 90)
    print(f"1. MaxSim (Unnormalized sum over query tokens):")
    print(f"   * Student -> Teacher fit:  y_teacher = {reg_max['slope_student_to_teacher']:.4f} * s_maxsim + ({reg_max['bias_student_to_teacher']:.4f})")
    print(f"   * Teacher -> Student fit:  s_maxsim  = {reg_max['slope_teacher_to_student']:.4f} * y_teacher + ({reg_max['bias_teacher_to_student']:.4f})")
    print(f"   * Pearson r: {reg_max['pearson_r']:.4f}, Spearman ρ: {reg_max['spearman_rho']:.4f}, R²: {reg_max['r2']:.4f}")
    print(f"   * RMSE Raw (s - y): {reg_max['rmse_raw']:.2f} -> RMSE After Affine Calib: {reg_max['rmse_calibrated']:.2f}")

    print(f"\n2. MeanMaxSim (Length-normalized mean over query tokens):")
    print(f"   * Student -> Teacher fit:  y_teacher     = {reg_mean['slope_student_to_teacher']:.4f} * s_meanmaxsim + ({reg_mean['bias_student_to_teacher']:.4f})")
    print(f"   * Teacher -> Student fit:  s_meanmaxsim = {reg_mean['slope_teacher_to_student']:.4f} * y_teacher + ({reg_mean['bias_teacher_to_student']:.4f})")
    print(f"   * Pearson r: {reg_mean['pearson_r']:.4f}, Spearman ρ: {reg_mean['spearman_rho']:.4f}, R²: {reg_mean['r2']:.4f}")
    print(f"   * RMSE Raw (s - y): {reg_mean['rmse_raw']:.2f} -> RMSE After Affine Calib: {reg_mean['rmse_calibrated']:.2f}")

    # Compare fitted teacher calibration to default rule (0.02 * y + 0.70)
    alpha_opt = reg_mean['slope_teacher_to_student']
    beta_opt = reg_mean['bias_teacher_to_student']
    rmse_opt = np.sqrt(np.mean((meanmaxsim - (alpha_opt * teachers + beta_opt)) ** 2))
    rmse_rule = np.sqrt(np.mean((meanmaxsim - (0.02 * teachers + 0.70)) ** 2))
    print(f"\n3. Evaluation of Teacher-to-MeanMaxSim Calibrations:")
    print(f"   * Optimal least-squares: y_scaled = {alpha_opt:.4f} * y + {beta_opt:.4f}  (RMSE vs LateOn = {rmse_opt:.4f})")
    print(f"   * Fixed rule:            y_scaled = 0.0200 * y + 0.7000  (RMSE vs LateOn = {rmse_rule:.4f})")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("GENERALIZED SIGMOID CALIBRATION RESULTS (Teacher Logits -> LateOn Scores)")
    print("=" * 90)
    if sig_mean.get("success"):
        print(f"1. MeanMaxSim Sigmoid Fit:")
        print(f"   * Formula: s(y) = {sig_mean['s_min']:.4f} + ({sig_mean['s_max']:.4f} - {sig_mean['s_min']:.4f}) / (1 + exp(-(y - {sig_mean['y0']:.4f}) / {sig_mean['tau']:.4f}))")
        print(f"   * Fitted: s_min={sig_mean['s_min']:.4f}, s_max={sig_mean['s_max']:.4f}, y0={sig_mean['y0']:.4f}, tau={sig_mean['tau']:.4f}")
        print(f"   * R²: {sig_mean['r2']:.4f}, RMSE: {sig_mean['rmse']:.4f} (vs Affine RMSE: {rmse_opt:.4f})")
    else:
        print(f"1. MeanMaxSim Sigmoid Fit: Failed ({sig_mean.get('error')})")

    if sig_max.get("success"):
        print(f"\n2. MaxSim Sigmoid Fit:")
        print(f"   * Formula: s(y) = {sig_max['s_min']:.4f} + ({sig_max['s_max']:.4f} - {sig_max['s_min']:.4f}) / (1 + exp(-(y - {sig_max['y0']:.4f}) / {sig_max['tau']:.4f}))")
        print(f"   * Fitted: s_min={sig_max['s_min']:.4f}, s_max={sig_max['s_max']:.4f}, y0={sig_max['y0']:.4f}, tau={sig_max['tau']:.4f}")
        print(f"   * R²: {sig_max['r2']:.4f}, RMSE: {sig_max['rmse']:.4f}")
    else:
        print(f"\n2. MaxSim Sigmoid Fit: Failed ({sig_max.get('error')})")
    print("=" * 90 + "\n")

    # 7. Generate Plot
    output_path = Path(args.output_path)
    plot_distributions(
        teachers, maxsim, meanmaxsim, reg_max, reg_mean, sig_max, sig_mean, output_path
    )
    logger.info("Done! Calibration analysis complete.")


if __name__ == "__main__":
    main()
