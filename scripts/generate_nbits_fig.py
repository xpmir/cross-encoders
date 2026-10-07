import argparse
import logging
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FONT_SIZE = 14

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": [""],
        "font.size": FONT_SIZE,
        "axes.labelsize": 11,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": FONT_SIZE - 3,
    }
)
plt.style.use("ggplot")

DEFAULT_HOST = "JZ"
DATA_FILE_PATH = "/lustre/fsn1/projects/rech/npl/commun/micev2/experiments/eval_lateon_plaid_abl_nbits/20261007_113734/results/results.csv"


def read_remote_csv(
    host: str,
    remote_path: str,
    user: str | None = None,
    port: int | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Copy a CSV from a remote host using scp, read it with pandas,
    and automatically remove the temporary local file.
    """
    local_candidate = Path(remote_path)
    if local_candidate.exists():
        logging.info("Path %s found locally, reading directly.", remote_path)
        return pd.read_csv(local_candidate, **read_csv_kwargs)

    # If already cached locally in current working dir, use it
    local_fallback = Path(local_candidate.name)
    if local_fallback.exists():
        logging.info("Found %s in current directory, reading directly.", local_fallback)
        return pd.read_csv(local_fallback, **read_csv_kwargs)

    if "@" in host:
        user, host = host.split("@", 1)
    remote = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(remote_path).name

        cmd = ["scp"]
        if port is not None:
            cmd += ["-P", str(port)]
        cmd += [remote, str(local_path)]

        logging.info("Fetching remote file via SCP: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error("Failed to copy %s via scp: %s", remote, e)
            raise

        logging.info("Reading CSV from temporary path: %s", local_path)
        df = pd.read_csv(local_path, **read_csv_kwargs)

    return df


def plot_nbits(
    df: pd.DataFrame,
    metric: str = "nDCG@10",
    output_path: str = "figures/ecir_nbits_ablation.png",
    show: bool = False,
) -> None:
    """Plot nbits vs normalized nDCG (perf - perf_nbits_8) for each dataset with lines/colors,
    including error bars for the mean across datasets.
    """
    valid_df = df[df["n_bits"].notna()].copy()
    if valid_df.empty:
        raise ValueError("No valid rows with 'n_bits' found in DataFrame.")

    if metric not in valid_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns: {list(valid_df.columns)}"
        )

    valid_df["n_bits"] = pd.to_numeric(valid_df["n_bits"])
    valid_df[metric] = pd.to_numeric(valid_df[metric])

    scorers = [s for s in valid_df["scorer"].dropna().unique()]
    datasets = [d for d in valid_df["dataset"].unique() if d != "mean"]

    # Distinct colors for each individual dataset
    colormap = plt.get_cmap("tab10")
    dataset_colors = {d: colormap(i) for i, d in enumerate(datasets)}

    n_scorers = len(scorers)
    fig, axes = plt.subplots(
        1,
        n_scorers,
        figsize=(7 * n_scorers, 5.5),
        sharey=True,
        dpi=150,
    )
    if n_scorers == 1:
        axes = [axes]

    for ax, scorer in zip(axes, scorers):
        scorer_df = valid_df[valid_df["scorer"] == scorer].copy()
        nbits_values = sorted(scorer_df["n_bits"].unique())

        norm_rows = []
        # Plot line for each dataset normalized to 8-bits
        for d in datasets:
            d_df = scorer_df[scorer_df["dataset"] == d].sort_values("n_bits").copy()
            if d_df.empty:
                continue
            ref_rows = d_df[d_df["n_bits"] == 8.0]
            if ref_rows.empty:
                continue
            ref_val = ref_rows[metric].iloc[0]
            d_df["norm_perf"] = d_df[metric] - ref_val
            norm_rows.append(d_df)

            dataset_label = d.replace("_decontaminated", "")
            ax.plot(
                d_df["n_bits"],
                d_df["norm_perf"],
                marker="o",
                label=dataset_label,
                color=dataset_colors[d],
                linewidth=1.8,
                markersize=6,
                alpha=0.9,
            )

        # Compute mean across datasets
        all_norm_df = pd.concat(norm_rows, ignore_index=True)
        agg = (
            all_norm_df.groupby("n_bits")["norm_perf"]
            .mean()
            .reset_index()
            .sort_values("n_bits")
        )

        ax.plot(
            agg["n_bits"],
            agg["norm_perf"],
            label="Mean",
            color="black",
            marker="s",
            linewidth=2.2,
            markersize=7,
            linestyle="--",
            zorder=5,
        )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        scorer_display_map = {
            "train_micev2_ettin68m_l10+4_v3_plaid_128": "Mice-ettin68m",
            "train_micev2_ettin150m_l12+4_v3_plaid_128": "Mice-ettin150m",
            "lightonai/LateOn": "LateOn",
        }
        first_stage_display_map = {
            "plaid": "PLAID",
            "naver/splade-v3-distilbert": "SPLADE-v3",
        }
        model_name = scorer_display_map.get(scorer, scorer.split("/")[-1])
        first_stage = scorer_df["first_stage"].dropna().unique()
        first_stage_name = first_stage_display_map.get(first_stage[0].lower(), first_stage[0]) if len(first_stage) > 0 else ""
        ax.set_title(f"{model_name}\n(first stage: {first_stage_name})", fontweight="bold", fontsize=FONT_SIZE)
        ax.set_xlabel("nbits (PLAID)")
        ax.set_xticks(nbits_values)
        ax.set_xticklabels([str(int(b)) if b.is_integer() else str(b) for b in nbits_values])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)

    axes[0].set_ylabel(f"Δ {metric} (relative to 8-bits)")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    logging.info("Saved plot to: %s", out.resolve())

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch results CSV and plot nbits ablation.")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Remote host to fetch CSV from (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DATA_FILE_PATH,
        help=f"Path to CSV on host (default: {DATA_FILE_PATH}).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="nDCG@10",
        help="Metric column to plot (default: 'nDCG@10').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/nbits_ablation.png",
        help="Output path for the generated plot (default: 'nbits_ablation.png').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window.",
    )
    args = parser.parse_args()

    df = read_remote_csv(host=args.host, remote_path=args.path)

    print("\n=== RESULTS DATAFRAME ===")
    print(df.to_string())

    plot_nbits(df, metric=args.metric, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
