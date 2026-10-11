import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter
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

FONT_SIZE = 14
FIG_SIZE = (8, 6)

SIZE_CLASSES = ["32m", "68m", "150m", "400m", "lateOn-unsup", "other"]

SIZE_COLORS = {
    "32m": "#f29f28",
    "68m": "#d6191b",
    "150m": "#221199",
    "400m": "#dfb428",
    "lateOn-unsup": "#16a085",
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

# Color for Pareto curves
pareto_color = "#27ae60"

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


def normalize_store_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes 'store' column to boolean (True/False)."""
    if "store" in df.columns:
        df["store"] = df["store"].apply(
            lambda v: True
            if (isinstance(v, bool) and v)
            or str(v).strip().lower() in ("true", "1", "yes")
            else False
        )
    else:
        df["store"] = False
    return df


def find_precomp_column(df: pd.DataFrame) -> str | None:
    """Finds the precomputed FLOPs column name if present."""
    for candidate in [
        "Flops_precomp",
        "Flops_precompute",
        "flops_precompute",
        "flops_precomp",
    ]:
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    return None


def add_effective_flops_column(
    df: pd.DataFrame,
    flops_col: str = "Flops",
    precomp_col: str | None = None,
) -> pd.DataFrame:
    """Computes an 'effective_flops' column taking precomputed FLOPs when store is True.

    For rows where store is True, uses precomputed FLOPs (Flops_precomp).
    Otherwise, uses the baseline FLOPs.
    """
    try:
        matched_flops = find_column(df, flops_col)
    except ValueError:
        matched_flops = None

    matched_precomp = (
        find_column(df, precomp_col) if precomp_col else find_precomp_column(df)
    )

    if matched_flops is None:
        logging.warning("Flops column '%s' not found, skipping 'effective_flops'.", flops_col)
        return df

    def get_effective(row: pd.Series) -> float | None:
        is_stored = bool(row.get("store", False))
        if is_stored:
            if matched_precomp is not None:
                val = pd.to_numeric(row[matched_precomp], errors="coerce")
                if pd.notna(val):
                    return float(val)
                logging.warning(
                    "Model '%s' has store=True but '%s' is missing/invalid (%s). "
                    "Falling back to '%s'.",
                    row.get("scorer", "unknown"),
                    matched_precomp,
                    row[matched_precomp],
                    matched_flops,
                )
            else:
                logging.warning(
                    "Model '%s' has store=True but no precompute FLOPs column found. "
                    "Falling back to '%s'.",
                    row.get("scorer", "unknown"),
                    matched_flops,
                )
        base_val = pd.to_numeric(row[matched_flops], errors="coerce")
        return float(base_val) if pd.notna(base_val) else None

    df["effective_flops"] = df.apply(get_effective, axis=1)
    logging.info(
        "Computed 'effective_flops' column (using '%s' when store=True, else '%s').",
        matched_precomp if matched_precomp else "N/A",
        matched_flops,
    )
    return df


def fetch_sheet_as_df(
    sheet_url: str,
    metric: str | None = "BEIR13 Decont",
    tok: str | None = "1024-1024",
    drop_empty_cols: bool = True,
    flops_col: str = "Flops",
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fetches a Google Sheet as a pandas DataFrame and applies filtering.

    Args:
        sheet_url: Google Sheets URL.
        metric: Column name to filter on. Rows with NaN in this column are removed.
        tok: Tokenizer configuration to filter on (e.g. '1024-1024', '32-512', or 'all').
        drop_empty_cols: Whether to drop columns containing only NaNs.
        flops_col: Flops column name for computing effective FLOPs.
        keep_cols: Optional list of column names that should never be dropped even if empty.
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

    # Normalize store column to boolean before empty column dropping
    df = normalize_store_column(df)

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
        protected = [c.lower() for c in (keep_cols or [])]
        all_nan_cols = [
            c for c in df.columns if df[c].isna().all() and c.lower() not in protected
        ]
        if all_nan_cols:
            logging.info("Dropping columns with 100%% NaNs: %s", all_nan_cols)
            df = df.drop(columns=all_nan_cols)

    df = add_size_class_columns(df)
    df = add_type_column(df)
    df = add_effective_flops_column(df, flops_col=flops_col)
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
    """Adds size class columns based on 'scorer' and 'base' columns.

    Adds boolean columns for '68m', '150m', '400m', 'lateOn-unsup', and a categorical 'size_class' column.
    """
    if "scorer" not in df.columns:
        logging.warning("'scorer' column not found, skipping size class detection.")
        return df

    def get_size(row: pd.Series) -> str:
        scorer_val = str(row.get("scorer", "")).lower()
        base_val = str(row.get("base", "")).lower()

        # Check for LateOn first
        if "lateon" in scorer_val or "lateon" in base_val:
            return "lateOn-unsup"

        for s in [scorer_val, base_val]:
            for size in ["32m", "68m", "150m", "400m"]:
                prefix = size[:-1]
                pattern = rf"{size}|(?<![0-9]){prefix}(?![0-9])"
                if re.search(pattern, s):
                    return size
        return "other"

    size_series = df.apply(get_size, axis=1)
    df["size_class"] = size_series

    for size in ["68m", "150m", "400m"]:
        df[size] = size_series == size
    df["lateOn-unsup"] = size_series == "lateOn-unsup"

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
    pareto_no_store_df: pd.DataFrame | None = None,
    y_break: tuple[float, float] | None = (0.48, 0.56),
) -> None:
    """Plots all valid points and draws the step-post Pareto frontier, with optional Y-axis break."""
    matched_x = find_column(df, x_col)
    matched_y = find_column(df, y_col)

    valid_df = df.dropna(subset=[matched_x, matched_y]).copy()
    valid_df[matched_x] = pd.to_numeric(valid_df[matched_x])
    valid_df[matched_y] = pd.to_numeric(valid_df[matched_y])

    # Convert Flops to GFLOPs if scale is large; format latency as ms/query
    is_latency = any(k in matched_x.lower() for k in ["latency", "ms", "time"])
    if is_latency:
        scale = 1.0
        x_display_name = "Latency (ms/query)"
        x_label = "Latency (ms/query)"
    else:
        use_gflops = valid_df[matched_x].max() > 1e6
        scale = 1e9 if use_gflops else 1.0
        x_display_name = "FLOPs" if matched_x in ["effective_flops", "Flops"] else matched_x
        x_label = f"{x_display_name} (GFLOPs)" if use_gflops else x_display_name

    # Determine if Y-axis break is applicable
    y_data_min = float(valid_df[matched_y].min())
    y_data_max = float(valid_df[matched_y].max())
    y_margin = 0.005
    y_plot_bottom = y_data_min - y_margin

    use_break = False
    if y_break is not None:
        b_low, b_high = y_break
        # Auto-scale break range if metric is 0-1 vs 0-100
        if y_data_max <= 1.0:
            if b_low > 1.0:
                b_low /= 100.0
            if b_high > 1.0:
                b_high /= 100.0
        else:
            if b_low <= 1.0:
                b_low *= 100.0
            if b_high <= 1.0:
                b_high *= 100.0

        if (valid_df[matched_y] < b_low).any() and (valid_df[matched_y] > b_high).any():
            use_break = True
            break_low, break_high = b_low, b_high
            logging.info("Applying Y-axis break between %g and %g", break_low, break_high)

    if use_break:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=FIG_SIZE,
            dpi=200,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
        axes = [ax_top, ax_bottom]
        primary_ax = ax_bottom
    else:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=200)
        axes = [ax]
        primary_ax = ax

    # Precompute frontier curve coordinates
    p_ns_coords: tuple[list[float], list[float]] | None = None
    if pareto_no_store_df is not None and not pareto_no_store_df.empty:
        p_ns_x = (pd.to_numeric(pareto_no_store_df[matched_x]) / scale).tolist()
        p_ns_y = pd.to_numeric(pareto_no_store_df[matched_y]).tolist()
        if p_ns_x and p_ns_y:
            p_ns_x = [p_ns_x[0]] + p_ns_x
            p_ns_y = [y_plot_bottom] + p_ns_y
        p_ns_coords = (p_ns_x, p_ns_y)

    p_coords: tuple[list[float], list[float]] | None = None
    if not pareto_df.empty:
        p_x = (pd.to_numeric(pareto_df[matched_x]) / scale).tolist()
        p_y = pd.to_numeric(pareto_df[matched_y]).tolist()
        if p_x and p_y:
            p_x = [p_x[0]] + p_x
            p_y = [y_plot_bottom] + p_y
        p_coords = (p_x, p_y)

    has_size = "size_class" in valid_df.columns
    has_type = "type" in valid_df.columns
    has_store = "store" in valid_df.columns
    adjust_text_kwargs = {
        "force_text": (0.09, 0.03),   # gentler collision repulsion
        "force_static": (0.05, 0.08),
        "force_pull": (0.05, 0.08),   # pull labels back toward their points
        "expand": (1.02, 1.04),       # modest label spacing
        "max_move": (3, 2),           # smaller adjustment steps
        "iter_lim": 100,
        "min_arrow_len": 0,
        "arrowprops": {
            "arrowstyle": "-",
            "color": "#777777",
            "lw": 0.6,
            "alpha": 0.7,
        },
    }

    # Render curves and scatter points across active axes
    for current_ax in axes:
        # 1. Plot Pareto frontiers first (behind scatter points)
        if p_ns_coords is not None:
            current_ax.plot(
                p_ns_coords[0],
                p_ns_coords[1],
                drawstyle="steps-post",
                color=pareto_color,
                linestyle="-",
                linewidth=1.8,
                zorder=2,
            )

        if p_coords is not None:
            current_ax.plot(
                p_coords[0],
                p_coords[1],
                drawstyle="steps-post",
                color=pareto_color,
                linestyle=":",
                linewidth=2.0,
                zorder=2,
            )

        # 2. Scatter points grouped by size_class, type, and store
        if has_size and has_type:
            group_cols = ["size_class", "type"]
            if has_store:
                group_cols.append("store")

            for group_keys, subset in valid_df.groupby(group_cols):
                if has_store:
                    size_cls, mtype, is_stored = group_keys
                else:
                    size_cls, mtype = group_keys
                    is_stored = False

                color = SIZE_COLORS.get(size_cls, "#7f8c8d")
                marker = TYPE_MARKERS.get(mtype, "o")
                marker_size = 140 if marker == "*" else 90

                if is_stored:
                    facecolor = "white"
                    edgecolor = color
                    linewidth = 1.0
                    alpha = 1.0
                else:
                    facecolor = color
                    edgecolor = "black"
                    linewidth = 1.0
                    alpha = 0.85

                current_ax.scatter(
                    subset[matched_x] / scale,
                    subset[matched_y],
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    marker=marker,
                    s=marker_size,
                    alpha=alpha,
                    zorder=4,
                )
        else:
            current_ax.scatter(
                valid_df[matched_x] / scale,
                valid_df[matched_y],
                color="#3498db",
                edgecolors="black",
                s=80,
                alpha=0.8,
                zorder=4,
            )

        current_ax.grid(True, linestyle="--", alpha=0.5)

    # Label all points on their respective axes
    if use_break:
        texts_top = []
        texts_bottom = []
        for _, row in valid_df.iterrows():
            lbl = str(row.get("scorer", "Model"))
            px_val = float(row[matched_x]) / scale
            py_val = float(row[matched_y])
            target_ax = ax_top if py_val >= break_high else ax_bottom
            txt = target_ax.text(
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
            if py_val >= break_high:
                texts_top.append(txt)
            else:
                texts_bottom.append(txt)

        try:
            from adjustText import adjust_text

            if texts_top:
                adjust_text(texts_top, ax=ax_top, **adjust_text_kwargs)
            if texts_bottom:
                adjust_text(texts_bottom, ax=ax_bottom, **adjust_text_kwargs)
        except Exception:
            pass
    else:
        texts = []
        for _, row in valid_df.iterrows():
            lbl = str(row.get("scorer", "Model"))
            px_val = float(row[matched_x]) / scale
            py_val = float(row[matched_y])
            txt = primary_ax.text(
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

            adjust_text(texts, ax=primary_ax, **adjust_text_kwargs)
        except Exception:
            pass

    # Configure axes limits and visual break marks
    if use_break:
        ax_top.set_ylim(bottom=break_high, top=y_data_max + y_margin)
        ax_bottom.set_ylim(bottom=y_plot_bottom, top=break_low)

        # Hide adjacent spines
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(top=False, labeltop=False)
        ax_bottom.xaxis.tick_bottom()

        # Add diagonal cut marks to indicate broken axis
        # Standard matplotlib pattern for broken y-axis
        d = 0.012
        kwargs = dict(color="k", clip_on=False, linewidth=1.2)
        # Top subplot bottom cut marks
        ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)

        # Bottom subplot top cut marks
        ax_bottom.plot((-d, +d), (1 - 3 * d, 1 + 3 * d), transform=ax_bottom.transAxes, **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - 3 * d, 1 + 3 * d), transform=ax_bottom.transAxes, **kwargs)

        ax_bottom.set_xlabel(x_label)
        # Centered Y label between the two panels
        fig.text(
            0.03,
            0.5,
            "Avg. nDCG@10 on BEIR13 (decontaminated)",
            va="center",
            rotation="vertical",
            fontsize=plt.rcParams["axes.labelsize"],
        )
    else:
        primary_ax.set_ylim(bottom=y_plot_bottom, top=y_data_max + y_margin)
        primary_ax.set_xlabel(x_label)
        primary_ax.set_ylabel("Avg. nDCG@10 on BEIR13 (decontaminated)")

    y_tick_formatter = FuncFormatter(lambda value, _: f"{100 * value:.1f}")
    for current_ax in axes:
        current_ax.yaxis.set_major_formatter(y_tick_formatter)

    # --- Assemble Custom Legend with Backbone, Type, Doc Precompute, and Pareto sections ---
    legend_handles = []

    # Section A: Backbone Architectures
    legend_handles.append(Line2D([0], [0], color="none", label="Backbone", markersize=0))
    if has_size:
        present_classes = [
            c
            for c in SIZE_CLASSES
            if (valid_df["size_class"] == c).any()
        ]
        for size_cls in present_classes:
            color = SIZE_COLORS.get(size_cls, "#7f8c8d")
            if size_cls in ["68m", "150m", "400m"]:
                label = f"ettin-{size_cls}"
            elif size_cls == "lateOn-unsup":
                label = "lateOn-unsup"
            else:
                label = size_cls
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
        present_types = [
            t
            for t in ["cross-encoder", "mice", "colbert", "other"]
            if (valid_df["type"] == t).any()
        ]
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

    # Section C: PLAID Store
    legend_handles.append(
        Line2D([0], [0], color="none", label="PLAID Store", markersize=0)
    )
    legend_handles.append(
        (
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="#333333",
                markeredgewidth=1.0,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#888888",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=8,
            ),
        )
    )
    legend_labels = [
        handle.get_label()
        if isinstance(handle, Line2D)
        else "Yes / No"
        for handle in legend_handles
    ]

    # Section D: Pareto Frontier
    if pareto_no_store_df is not None and not pareto_no_store_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=pareto_color,
                linestyle="-",
                linewidth=1.8,
                label="Pareto (no precompute)",
            )
        )
    if not pareto_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=pareto_color,
                linestyle=":",
                linewidth=2.0,
                label="Pareto (w/ precompute)",
            )
        )

    leg = primary_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="lightgray",
    )
    if leg:
        for text in leg.get_texts():
            if text.get_text() in ["Backbone", "Type", "PLAID Store"]:
                text.set_fontweight("bold")
                text.set_color("#333333")

    if not use_break:
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
    """Formats DataFrame into a LaTeX table keeping scorer, BEIR datasets, and LoTTE datasets.

    BEIR datasets are mapped to abbreviations using DATASET_TO_ABB and sorted
    alphabetically. LoTTE sub-datasets and Lotte-S aggregate are appended before Avg.
    The aggregation column is placed as the last column.
    """
    if "scorer" not in df.columns:
        raise ValueError("DataFrame must contain 'scorer' column.")

    # Identify individual BEIR and LoTTE columns using DATASET_TO_ABB
    lotte_keys = {
        "lotte_lifestyle",
        "lotte_recreation",
        "lotte_science",
        "lotte_technology",
        "lotte_writing",
    }

    beir_cols: list[tuple[str, str]] = []
    lotte_cols: list[tuple[str, str]] = []

    for c in df.columns:
        if c in ["base", "scorer", aggregation_col, "Lotte-S"]:
            continue
        norm = c.replace(" ", "_").lower()
        if norm in lotte_keys:
            lotte_cols.append((c, DATASET_TO_ABB.get(norm, c)))
        elif c.lower() in lotte_keys:
            lotte_cols.append((c, DATASET_TO_ABB.get(c.lower(), c)))
        elif norm in DATASET_TO_ABB:
            beir_cols.append((c, DATASET_TO_ABB[norm]))
        elif c.lower() in DATASET_TO_ABB:
            beir_cols.append((c, DATASET_TO_ABB[c.lower()]))

    # Sort BEIR columns alphabetically by abbreviation
    beir_cols.sort(key=lambda x: x[1])

    # Sort LoTTE detail columns alphabetically by abbreviation
    lotte_cols.sort(key=lambda x: x[1])

    # Check for Lotte-S aggregate column
    lotte_s_col = next((c for c in df.columns if c.lower() in ["lotte-s", "lotte_s"]), None)

    # Build selected column list: scorer, optional 1st stage, optional store, sorted BEIR, LoTTE details, Lotte-S, and aggregation
    first_stage_col = next(
        (c for c in df.columns if c.lower() in ["1st stage", "first stage", "1st_stage", "1st st"]),
        None,
    )
    has_first_stage = first_stage_col is not None
    has_store = "store" in df.columns

    selected_cols = ["scorer"]
    rename_map = {"scorer": "Scorer"}

    if has_first_stage:
        selected_cols.append(first_stage_col)
        rename_map[first_stage_col] = "1st St"

    if has_store:
        selected_cols.append("store")
        rename_map["store"] = "Store"

    # BEIR columns
    selected_cols.extend([c for c, _ in beir_cols])
    for c, abb in beir_cols:
        rename_map[c] = abb

    # LoTTE detail columns
    selected_cols.extend([c for c, _ in lotte_cols])
    for c, abb in lotte_cols:
        rename_map[c] = abb

    # Lotte-S column
    if lotte_s_col is not None:
        selected_cols.append(lotte_s_col)
        rename_map[lotte_s_col] = "Lotte-S"

    has_agg = aggregation_col in df.columns
    if has_agg:
        selected_cols.append(aggregation_col)
        rename_map[aggregation_col] = aggregation_label

    latex_df = df[selected_cols].copy()

    # Escape LaTeX special characters in Scorer
    latex_df["scorer"] = latex_df["scorer"].apply(escape_latex)

    # Format 1st stage column: "S" for splade, "P" for plaid
    if has_first_stage:
        def format_1st_stage(val: object) -> str:
            if not isinstance(val, str) or pd.isna(val):
                return "-"
            v = val.lower()
            if "splade" in v:
                return "S"
            if "plaid" in v:
                return "P"
            return escape_latex(str(val))

        latex_df[first_stage_col] = latex_df[first_stage_col].apply(format_1st_stage)

    # Format store column with \vmark or \xmark
    if has_store:
        latex_df["store"] = latex_df["store"].apply(
            lambda v: r"\vmark" if bool(v) else r"\xmark"
        )

    # Format numeric columns as 100 * x with 1 decimal place
    all_numeric_dataset_cols = [c for c, _ in beir_cols] + [c for c, _ in lotte_cols]
    if lotte_s_col is not None:
        all_numeric_dataset_cols.append(lotte_s_col)

    for col in all_numeric_dataset_cols:
        latex_df[col] = pd.to_numeric(latex_df[col], errors="coerce").apply(
            lambda v: f"{100 * v:.1f}" if pd.notna(v) else "-"
        )
    if has_agg:
        latex_df[aggregation_col] = pd.to_numeric(
            latex_df[aggregation_col], errors="coerce"
        ).apply(lambda v: f"{100 * v:.1f}" if pd.notna(v) else "-")

    latex_df = latex_df.rename(columns=rename_map)

    num_data_cols = len(beir_cols) + len(lotte_cols) + (1 if lotte_s_col is not None else 0) + (1 if has_agg else 0)
    column_format = (
        "l"
        + ("c" if has_first_stage else "")
        + ("c" if has_store else "")
        + "c" * num_data_cols
    )
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
        "--x-col",
        type=str,
        default="latency_ms_per_query",
        help="Column name for X axis (e.g. 'latency_ms_per_query', 'Flops', 'effective_flops'). Default: 'latency_ms_per_query'.",
    )
    parser.add_argument(
        "--flops-col",
        type=str,
        default="Flops",
        help="Column name for Flops in X (fallback/alternate). Default: 'Flops'.",
    )
    parser.add_argument(
        "--keep-empty-cols",
        action="store_true",
        help="Keep columns that are completely NaN.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        default=None,
        help="Optional file path to save the Pareto plot (e.g. pareto.png). If specified without value, auto-generates a filename with metric and x-col.",
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
    parser.add_argument(
        "--y-break",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to insert a broken Y-axis between 48 and 56 (0.48 and 0.56) if data spans across. Default: True.",
    )
    parser.add_argument(
        "--y-break-range",
        type=float,
        nargs=2,
        default=[0.48, 0.56],
        metavar=("LOW", "HIGH"),
        help="Y-axis break range (default: 0.48 0.56).",
    )
    args = parser.parse_args()

    # Determine x_col and dataset columns to protect and use
    requested_x = args.x_col
    lotte_cols = [
        "lotte_lifestyle",
        "lotte_recreation",
        "lotte_science",
        "lotte_technology",
        "lotte_writing",
        "Lotte-S",
    ]
    keep_cols = [requested_x, args.flops_col] + lotte_cols

    df = fetch_sheet_as_df(
        args.url,
        metric=args.metric,
        tok=args.tok,
        drop_empty_cols=not args.keep_empty_cols,
        flops_col=args.flops_col,
        keep_cols=keep_cols,
    )

    print("\n=== FILTERED DATAFRAME ===")
    if args.head:
        print(df.head(args.head).to_string())
    else:
        print(df.to_string())

    # Resolve pareto x column
    matched_x_col = None
    try:
        matched_x_col = find_column(df, requested_x)
    except ValueError:
        # If requested x_col was Flops/effective_flops or missing
        if requested_x.lower() in ["flops", "effective_flops"]:
            matched_x_col = (
                "effective_flops"
                if "effective_flops" in df.columns
                else args.flops_col
            )
        else:
            logging.warning(
                "Requested X column '%s' not found. Available: %s. Falling back to effective_flops.",
                requested_x,
                list(df.columns),
            )
            matched_x_col = (
                "effective_flops"
                if "effective_flops" in df.columns
                else args.flops_col
            )

    pareto_x_col = matched_x_col

    # Compute overall Pareto frontier (including precomputed doc reps)
    pareto_df = compute_pareto_front(df, x_col=pareto_x_col, y_col=args.metric)

    # Compute Pareto frontier strictly for models without document precomputation
    df_no_store = df[~df["store"]].copy() if "store" in df.columns else df.copy()
    pareto_no_store_df = compute_pareto_front(
        df_no_store, x_col=pareto_x_col, y_col=args.metric
    )

    display_cols = [
        col
        for col in [
            "scorer",
            "type",
            "size_class",
            "store",
            "base",
            "tok",
            pareto_x_col,
            args.flops_col,
            "Flops_precomp",
            "effective_flops",
            args.metric,
        ]
        if col in df.columns
    ]

    print(
        f"\n=== PARETO FRONTIER (OVERALL / W/ PRECOMPUTE, X: {pareto_x_col}, Y: {args.metric}) ==="
    )
    if not pareto_df.empty:
        print(
            pareto_df[[c for c in display_cols if c in pareto_df.columns]].to_string()
        )
    else:
        print("No Pareto-optimal points found (insufficient valid data).")

    print(
        f"\n=== PARETO FRONTIER (NO DOC PRECOMPUTE, X: {pareto_x_col}, Y: {args.metric}) ==="
    )
    if not pareto_no_store_df.empty:
        print(
            pareto_no_store_df[
                [c for c in display_cols if c in pareto_no_store_df.columns]
            ].to_string()
        )
    else:
        print("No Pareto-optimal points found (insufficient valid data).")

    if args.plot is not None:
        plot_path = args.plot
        if not plot_path:
            tok_slug = f"_{args.tok}" if args.tok else ""
            x_slug = "latency" if "lat" in pareto_x_col.lower() else "flops"
            metric_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.metric).strip("_").lower()
            plot_path = f"pareto_{metric_slug}_{x_slug}{tok_slug}.png"

        y_break_val = tuple(args.y_break_range) if args.y_break else None
        plot_pareto_frontier(
            df=df,
            pareto_df=pareto_df,
            x_col=pareto_x_col,
            y_col=args.metric,
            output_path=plot_path,
            pareto_no_store_df=pareto_no_store_df,
            y_break=y_break_val,
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
