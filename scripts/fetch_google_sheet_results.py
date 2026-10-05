import argparse
import logging
import re

import pandas as pd

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


def fetch_sheet_as_df(
    sheet_url: str,
    metric: str | None = None,
    drop_empty_cols: bool = True,
) -> pd.DataFrame:
    """Fetches a Google Sheet as a pandas DataFrame via its CSV export endpoint.

    Args:
        sheet_url: Google Sheets URL.
        metric: Column name to filter on. If specified, rows with NaN in this column are removed.
        drop_empty_cols: Whether to drop columns that contain only NaNs.
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

    if metric is not None:
        # Match metric name exactly or case-insensitively
        matched_col = None
        for col in df.columns:
            if col.lower() == metric.lower():
                matched_col = col
                break

        if matched_col is None:
            available_cols = ", ".join(f"'{c}'" for c in df.columns)
            error_msg = f"Metric '{metric}' not found in DataFrame columns. Available columns: {available_cols}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        before_rows = len(df)
        df = df.dropna(subset=[matched_col]).reset_index(drop=True)
        dropped_rows = before_rows - len(df)
        logging.info(
            "Filtered on metric '%s': retained %d rows (dropped %d rows with NaN)",
            matched_col,
            len(df),
            dropped_rows,
        )

    if drop_empty_cols:
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            logging.info("Dropping columns with 100%% NaNs: %s", all_nan_cols)
            df = df.drop(columns=all_nan_cols)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve a Google Sheet as a pandas DataFrame and print it."
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
        help="Metric column name to filter by (removes rows with NaNs in this column). Default: 'BEIR13 Decont'.",
    )
    parser.add_argument(
        "--keep-empty-cols",
        action="store_true",
        help="Keep columns that are completely NaN.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Number of rows to print (default prints all rows)",
    )
    args = parser.parse_args()

    df = fetch_sheet_as_df(
        args.url,
        metric=args.metric,
        drop_empty_cols=not args.keep_empty_cols,
    )

    if args.head:
        print(df.head(args.head).to_string())
    else:
        print(df.to_string())


if __name__ == "__main__":
    main()
