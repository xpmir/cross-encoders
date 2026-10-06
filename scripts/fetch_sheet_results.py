import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Ensure src is in sys.path for format imports
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

try:
    from format import DATASET_TO_ABB, escape_latex
except ImportError:
    from src.format import DATASET_TO_ABB, escape_latex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "103NSCbgBucaat2M8eQS7nrPYuFuBNIZKWme4ASNkUk0/edit?gid=1470966267#gid=1470966267"
)


def parse_google_sheet_url(url: str) -> tuple[str, str | None]:
    """Extracts the spreadsheet ID and gid from a Google Sheets URL."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError(f"Could not find a valid Google Spreadsheet ID in URL: {url}")
    sheet_id = match.group(1)

    gid_match = re.search(r"[?&#]gid=([0-9]+)", url)
    gid = gid_match.group(1) if gid_match else None

    return sheet_id, gid


def normalize_column_name(col: str) -> str:
    """Replaces newlines and consecutive whitespace with a single space."""
    return re.sub(r"\s+", " ", col).strip()


def find_column(df: pd.DataFrame, target_name: str) -> str:
    """Finds matching column name case-insensitively, or raises ValueError."""
    for col in df.columns:
        if col.lower() == target_name.lower():
            return col
    available_cols = ", ".join(f"'{c}'" for c in df.columns)
    error_msg = (
        f"Column '{target_name}' not found in DataFrame columns. "
        f"Available columns: {available_cols}"
    )
    logging.error(error_msg)
    raise ValueError(error_msg)


def fetch_sheet_as_df(
    sheet_url: str,
    metric: str | None = "BEIR13 Decont",
    tok: str | None = "1024-1024",
    drop_empty_cols: bool = True,
) -> pd.DataFrame:
    """Fetches a Google Sheet as a pandas DataFrame and applies filtering.

    Args:
        sheet_url: Google Sheets URL.
        metric: Column name to filter on. Rows with NaN in this column are removed.
        tok: Tokenizer configuration to filter on (e.g. '1024-1024', '32-512', or 'all').
        drop_empty_cols: Whether to drop columns containing only NaNs.
    """
    sheet_id, gid = parse_google_sheet_url(sheet_url)
    logging.info("Parsed Sheet ID: %s, gid: %s", sheet_id, gid)

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid is not None:
        export_url += f"&gid={gid}"

    logging.info("Fetching CSV from: %s", export_url)
    try:
        df = pd.read_csv(export_url)
        logging.info("Successfully fetched DataFrame: initial shape %s", df.shape)
    except Exception as e:
        logging.error("Failed to fetch or parse sheet from %s: %s", export_url, e)
        raise

    # Normalize column names (e.g. replace internal newlines from multi-line headers with space)
    df.columns = [normalize_column_name(str(c)) for c in df.columns]

    # Filter by tok if specified
    if tok and tok.lower() != "all":
        if "tok" not in df.columns:
            raise ValueError(f"'tok' column not found in columns: {list(df.columns)}")
        valid_tok_values = df["tok"].dropna().unique().tolist()
        if tok not in valid_tok_values:
            logging.warning(
                "Specified tok '%s' not among present values: %s",
                tok,
                valid_tok_values,
            )
        before_tok = len(df)
        df = df[df["tok"] == tok].reset_index(drop=True)
        logging.info(
            "Filtered on tok='%s': retained %d rows (dropped %d rows)",
            tok,
            len(df),
            before_tok - len(df),
        )

    # Filter by metric
    if metric is not None:
        matched_metric = find_column(df, metric)
        before_metric = len(df)
        df = df.dropna(subset=[matched_metric]).reset_index(drop=True)
        logging.info(
            "Filtered on metric '%s': retained %d rows (dropped %d rows with NaN)",
            matched_metric,
            len(df),
            before_metric - len(df),
        )

    if drop_empty_cols:
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            logging.info("Dropping columns with 100%% NaNs: %s", all_nan_cols)
            df = df.drop(columns=all_nan_cols)

    df = add_size_class_columns(df)
    df = add_type_column(df)
    return df


def add_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'type' column detecting 'colbert' (including 'lateon'), 'mice', and 'cross-encoder' in scorer."""
    if "scorer" not in df.columns:
        logging.warning("'scorer' column not found, skipping type detection.")
        return df

    def get_type(scorer_val: object) -> str:
        if not isinstance(scorer_val, str):
            return "other"
        s = scorer_val.lower()
        if "colbert" in s or "lateon" in s:
            return "colbert"
        if "mice" in s:
            return "mice"
        if (
            "cross-encoder" in s
            or "cross" in s
            or "reranker" in s
            or "baseline" in s
            or s.startswith("ce")
        ):
            return "cross-encoder"
        return "other"

    df["type"] = df["scorer"].apply(get_type)
    logging.info("Identified model types: %s", dict(df["type"].value_counts()))
    return df


def add_size_class_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds size class columns based on the 'scorer' column.

    Adds boolean columns for '68m', '150m', '400m', and a categorical 'size_class' column.
    """
    if "scorer" not in df.columns:
        logging.warning("'scorer' column not found, skipping size class detection.")
        return df

    def get_size(scorer_val: object) -> str | None:
        if not isinstance(scorer_val, str):
            return None
        s = scorer_val.lower()
        for size in ["68m", "150m", "400m"]:
            prefix = size[:-1]
            pattern = rf"{size}|(?<![0-9]){prefix}(?![0-9])"
            if re.search(pattern, s):
                return size
        return None

    size_series = df["scorer"].apply(get_size)
    df["size_class"] = size_series.fillna("other")

    for size in ["68m", "150m", "400m"]:
        df[size] = size_series == size

    logging.info(
        "Identified size classes: %s",
        dict(df["size_class"].value_counts()),
    )
    return df


def compute_pareto_front(
    df: pd.DataFrame,
    x_col: str = "Flops",
    y_col: str = "BEIR13 Decont",
) -> pd.DataFrame:
    """Computes the non-dominated Pareto frontier rows from df.

    Minimizes X (Flops) and maximizes Y (Metric).
    A point (x, y) is on the frontier if no other point has x' <= x and y' >= y
    with at least one strict inequality.

    Returns:
        pd.DataFrame containing only the Pareto-optimal rows, sorted by x_col ascending.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty, returning empty Pareto frontier.")
        return df.copy()

    matched_x = find_column(df, x_col)
    matched_y = find_column(df, y_col)

    work_df = df.copy()
    work_df[matched_x] = pd.to_numeric(work_df[matched_x], errors="coerce")
    work_df[matched_y] = pd.to_numeric(work_df[matched_y], errors="coerce")

    # Check for missing values in x_col
    missing_x = work_df[work_df[matched_x].isna()]
    if not missing_x.empty:
        scorers = missing_x.get("scorer", missing_x.index).tolist()
        logging.warning(
            "Excluding %d row(s) from Pareto front due to missing '%s': %s",
            len(missing_x),
            matched_x,
            scorers,
        )

    valid_df = work_df.dropna(subset=[matched_x, matched_y]).copy()
    if valid_df.empty:
        logging.warning(
            "No valid data points with both '%s' and '%s' for Pareto frontier.",
            matched_x,
            matched_y,
        )
        return valid_df

    # Sort by X ascending (lower compute is better) then Y descending (higher metric is better)
    sorted_df = valid_df.sort_values(by=[matched_x, matched_y], ascending=[True, False])

    pareto_indices: list[int] = []
    max_y_seen = -float("inf")

    for idx, row in sorted_df.iterrows():
        # A point is on the frontier if it yields strictly higher performance
        # than any alternative with less or equal compute.
        if row[matched_y] > max_y_seen:
            pareto_indices.append(idx)
            max_y_seen = row[matched_y]

    pareto_df = df.loc[pareto_indices].sort_values(by=matched_x).reset_index(drop=True)
    logging.info(
        "Pareto frontier identified %d non-dominated model(s) out of %d valid points.",
        len(pareto_df),
        len(valid_df),
    )
    return pareto_df


def get_pareto_frontier(
    group_df: pd.DataFrame,
    x_col: str = "Flops",
    y_col: str = "BEIR13 Decont",
    include_origin: bool = False,
) -> tuple[list[float], list[float]]:
    """Returns sorted coordinates (frontier_x, frontier_y) for the non-dominated Pareto frontier.

    Follows the convention from MiceV2Results.ipynb.
    """
    pareto_df = compute_pareto_front(group_df, x_col=x_col, y_col=y_col)
    if pareto_df.empty:
        return [], []

    matched_x = find_column(pareto_df, x_col)
    matched_y = find_column(pareto_df, y_col)

    fx = pd.to_numeric(pareto_df[matched_x]).tolist()
    fy = pd.to_numeric(pareto_df[matched_y]).tolist()

    if include_origin:
        fx = [0.0] + fx
        fy = [0.0] + fy

    return fx, fy


def plot_pareto_frontier(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str,
    title: str | None = None,
) -> None:
    """Plots all valid points and draws the step-post Pareto frontier."""
    matched_x = find_column(df, x_col)
    matched_y = find_column(df, y_col)

    valid_df = df.dropna(subset=[matched_x, matched_y]).copy()
    valid_df[matched_x] = pd.to_numeric(valid_df[matched_x])
    valid_df[matched_y] = pd.to_numeric(valid_df[matched_y])

    # Convert Flops to GFLOPs if scale is large
    use_gflops = valid_df[matched_x].max() > 1e6
    scale = 1e9 if use_gflops else 1.0
    x_label = f"{matched_x} (GFLOPs)" if use_gflops else matched_x

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    SIZE_COLORS = {
        "68m": "#f29f28",
        "150m": "#d6191b",
        "400m": "#221199",
        "other": "#7f8c8d",
    }

    TYPE_MARKERS = {
        "cross-encoder": "o",
        "mice": "*",
        "colbert": "^",
        "other": "s",
    }

    TYPE_LABELS = {
        "cross-encoder": "Cross-Encoder",
        "mice": "MICE",
        "colbert": "ColBERT",
        "other": "Other",
    }

    # Scatter points grouped by size_class and type
    has_size = "size_class" in valid_df.columns
    has_type = "type" in valid_df.columns

    if has_size and has_type:
        for (size_cls, mtype), subset in valid_df.groupby(["size_class", "type"]):
            color = SIZE_COLORS.get(size_cls, "#7f8c8d")
            marker = TYPE_MARKERS.get(mtype, "o")
            marker_size = 140 if marker == "*" else 90
            ax.scatter(
                subset[matched_x] / scale,
                subset[matched_y],
                color=color,
                marker=marker,
                edgecolors="black",
                s=marker_size,
                alpha=0.85,
                zorder=3,
            )
    else:
        ax.scatter(
            valid_df[matched_x] / scale,
            valid_df[matched_y],
            color="#3498db",
            edgecolors="black",
            s=80,
            alpha=0.8,
            zorder=3,
        )

    # Label all points
    texts = []
    for _, row in valid_df.iterrows():
        lbl = str(row.get("scorer", "Model"))
        px_val = float(row[matched_x]) / scale
        py_val = float(row[matched_y])
        txt = ax.text(
            px_val,
            py_val,
            f" {lbl}",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.6,
                edgecolor="none",
            ),
        )
        texts.append(txt)

    try:
        from adjustText import adjust_text

        adjust_text(texts, ax=ax)
    except Exception:
        pass

    # Draw frontier strictly as a dotted line without separate markers
    if not pareto_df.empty:
        p_x = (pd.to_numeric(pareto_df[matched_x]) / scale).tolist()
        p_y = pd.to_numeric(pareto_df[matched_y]).tolist()

        ax.plot(
            p_x,
            p_y,
            drawstyle="steps-post",
            color="#2c3e50",
            linestyle=":",
            linewidth=2.0,
            zorder=4,
        )

    # --- Assemble Custom Legend with Backbone and Type sections ---
    legend_handles = []

    # Section A: Backbone Architectures
    legend_handles.append(Line2D([0], [0], color="none", label="Backbone", markersize=0))
    if has_size:
        present_classes = [c for c in ["68m", "150m", "400m", "other"] if (valid_df["size_class"] == c).any()]
        for size_cls in present_classes:
            color = SIZE_COLORS.get(size_cls, "#7f8c8d")
            label = f"ettin-{size_cls}" if size_cls in ["68m", "150m", "400m"] else size_cls
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                    label=label,
                )
            )

    # Section B: Model Types & Family Shapes
    legend_handles.append(Line2D([0], [0], color="none", label="Type", markersize=0))
    if has_type:
        present_types = [t for t in ["cross-encoder", "mice", "colbert", "other"] if (valid_df["type"] == t).any()]
        for mtype in present_types:
            marker = TYPE_MARKERS.get(mtype, "o")
            label = TYPE_LABELS.get(mtype, mtype)
            marker_size = 11 if marker == "*" else 8
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor="#888888",
                    markeredgecolor="black",
                    markersize=marker_size,
                    label=label,
                )
            )

    # Section C: Pareto Frontier
    if not pareto_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#2c3e50",
                linestyle=":",
                linewidth=2.0,
                label="Pareto Frontier",
            )
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(matched_y)
    ax.set_title(title or f"Pareto Frontier: {matched_y} vs {x_label}")
    ax.grid(True, linestyle="--", alpha=0.5)

    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="lightgray",
    )
    if leg:
        for text in leg.get_texts():
            if text.get_text() in ["Backbone", "Type"]:
                text.set_fontweight("bold")
                text.set_color("#333333")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved Pareto plot to: %s", output_path)


def format_beir_latex_table(
    df: pd.DataFrame,
    aggregation_col: str = "BEIR13 Decont",
    aggregation_label: str = "Avg",
) -> str:
    """Formats DataFrame into a LaTeX table keeping only scorer and BEIR datasets.

    BEIR datasets are mapped to abbreviations using DATASET_TO_ABB and sorted
    alphabetically. The aggregation column is placed as the last column.
    """
    if "scorer" not in df.columns:
        raise ValueError("DataFrame must contain 'scorer' column.")

    # Identify individual BEIR columns using DATASET_TO_ABB
    beir_cols: list[tuple[str, str]] = []
    for c in df.columns:
        if c in ["base", "scorer", aggregation_col]:
            continue
        norm = c.replace(" ", "_").lower()
        if norm in DATASET_TO_ABB:
            beir_cols.append((c, DATASET_TO_ABB[norm]))
        elif c.lower() in DATASET_TO_ABB:
            beir_cols.append((c, DATASET_TO_ABB[c.lower()]))

    # Sort individual BEIR columns alphabetically by abbreviation
    beir_cols.sort(key=lambda x: x[1])

    # Build selected column list: scorer, sorted BEIR datasets, and aggregation
    selected_cols = ["scorer"] + [c for c, _ in beir_cols]
    rename_map = {"scorer": "Scorer"}
    for c, abb in beir_cols:
        rename_map[c] = abb

    has_agg = aggregation_col in df.columns
    if has_agg:
        selected_cols.append(aggregation_col)
        rename_map[aggregation_col] = aggregation_label

    latex_df = df[selected_cols].copy()

    # Escape LaTeX special characters in Scorer
    latex_df["scorer"] = latex_df["scorer"].apply(escape_latex)

    # Format numeric columns as 100 * x with 1 decimal place
    for col, _ in beir_cols:
        latex_df[col] = pd.to_numeric(latex_df[col], errors="coerce").apply(
            lambda v: f"{100 * v:.1f}" if pd.notna(v) else "-"
        )
    if has_agg:
        latex_df[aggregation_col] = pd.to_numeric(
            latex_df[aggregation_col], errors="coerce"
        ).apply(lambda v: f"{100 * v:.1f}" if pd.notna(v) else "-")

    latex_df = latex_df.rename(columns=rename_map)

    column_format = "l" + "c" * (len(beir_cols) + (1 if has_agg else 0))
    return latex_df.to_latex(index=False, column_format=column_format)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve a Google Sheet as a pandas DataFrame and compute the Pareto frontier."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_URL,
        help="Google Sheets URL (defaults to the provided sheet URL)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="BEIR13 Decont",
        help="Metric column name in Y (removes rows with NaNs in this column). Default: 'BEIR13 Decont'.",
    )
    parser.add_argument(
        "--tok",
        type=str,
        default="1024-1024",
        help="Tokenizer filter (e.g. '1024-1024', '32-512', or 'all' to disable). Default: '1024-1024'.",
    )
    parser.add_argument(
        "--flops-col",
        type=str,
        default="Flops",
        help="Column name for Flops in X. Default: 'Flops'.",
    )
    parser.add_argument(
        "--keep-empty-cols",
        action="store_true",
        help="Keep columns that are completely NaN.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional file path to save the Pareto plot (e.g. pareto.png or pareto.pdf).",
    )
    parser.add_argument(
        "--latex",
        nargs="?",
        const="-",
        default=None,
        help="Export results as a LaTeX table. Specify a file path to save, or use '-' to print to stdout.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Number of rows to print for the filtered table (default prints all rows)",
    )
    args = parser.parse_args()

    df = fetch_sheet_as_df(
        args.url,
        metric=args.metric,
        tok=args.tok,
        drop_empty_cols=not args.keep_empty_cols,
    )

    print("\n=== FILTERED DATAFRAME ===")
    if args.head:
        print(df.head(args.head).to_string())
    else:
        print(df.to_string())

    # Compute Pareto frontier with all rows
    pareto_df = compute_pareto_front(df, x_col=args.flops_col, y_col=args.metric)

    print(f"\n=== PARETO FRONTIER (X: {args.flops_col}, Y: {args.metric}) ===")
    display_cols = [
        col
        for col in ["scorer", "type", "size_class", "base", "tok", args.flops_col, args.metric]
        if col in pareto_df.columns
    ]
    if not pareto_df.empty:
        print(pareto_df[display_cols].to_string())
    else:
        print("No Pareto-optimal points found (insufficient valid data).")

    if args.plot:
        plot_pareto_frontier(
            df=df,
            pareto_df=pareto_df,
            x_col=args.flops_col,
            y_col=args.metric,
            output_path=args.plot,
        )

    if args.latex:
        latex_code = format_beir_latex_table(df, aggregation_col=args.metric)
        if args.latex == "-":
            print("\n=== LATEX TABLE ===")
            print(latex_code)
        else:
            out_file = Path(args.latex)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(latex_code)
            logging.info("Saved LaTeX table to: %s", out_file)


if __name__ == "__main__":
    main()
