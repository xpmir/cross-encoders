"""Utilities for extracting evaluation query latency from Experimaestro task logs."""

import logging
from pathlib import Path
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Matches tqdm progress bar lines, e.g.:
# 1373/1373 [00:26<00:00, 51.68queries/s]
# 61/61 [03:25<00:00,  3.37s/query]
# 349/349 [01:20<00:00, 4.35query/s]
TQDM_REGEX = re.compile(r"(\d+)/(\d+)\s*\[([0-9:]+)<[0-9:]+,\s*([^\]]+)\]")

# Matches explicit Stage 1 / Stage 2 / Full pipeline timing logs
STAGE1_REGEX = re.compile(
    r"Stage 1 retrieval total time.*?\((\d+(?:\.\d+)?)\s*ms/query", re.IGNORECASE
)
STAGE2_REGEX = re.compile(
    r"Stage 2 reranking total time.*?\((\d+(?:\.\d+)?)\s*ms/query", re.IGNORECASE
)
PIPELINE_REGEX = re.compile(
    r"Full two-stage pipeline completed in.*?\((\d+(?:\.\d+)?)\s*ms/query",
    re.IGNORECASE,
)


def parse_tqdm_time_seconds(time_str: str) -> float:
    """Parse MM:SS or HH:MM:SS into seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(parts[0])


def parse_tqdm_rate_to_ms(rate_str: str) -> Optional[float]:
    """Parse tqdm rate token into latency in milliseconds per query.
    Supports s/query, ms/query, queries/s, query/s, it/s, s/it, ms/it, etc.
    """
    rate_str = rate_str.strip()

    # Check s/query or s/it
    m_sec = re.match(r"^([\d.]+)\s*s/(?:query|queries|it)$", rate_str)
    if m_sec:
        sec = float(m_sec.group(1))
        return sec * 1000.0

    # Check ms/query or ms/it
    m_ms = re.match(r"^([\d.]+)\s*ms/(?:query|queries|it)$", rate_str)
    if m_ms:
        return float(m_ms.group(1))

    # Check queries/s, query/s, it/s
    m_qps = re.match(r"^([\d.]+)\s*(?:queries|query|it)/s$", rate_str)
    if m_qps:
        qps = float(m_qps.group(1))
        return (1000.0 / qps) if qps > 0 else 0.0

    return None


def parse_stage_timing_logs(content: str) -> Dict[str, float]:
    """Extract Stage 1, Stage 2, and pipeline latency metrics from structured log lines."""
    metrics: Dict[str, float] = {}

    m_pipe = PIPELINE_REGEX.findall(content)
    if m_pipe:
        metrics["latency_ms_per_query"] = float(m_pipe[-1])

    m_s1 = STAGE1_REGEX.findall(content)
    if m_s1:
        metrics["stage1_ms_per_query"] = float(m_s1[-1])

    m_s2 = STAGE2_REGEX.findall(content)
    if m_s2:
        metrics["stage2_ms_per_query"] = float(m_s2[-1])

    return metrics


def extract_latency_from_log_content(content: str) -> Dict[str, float]:
    """Extract query latency metrics from log content, prioritizing structured timing logs
    and falling back to final completed tqdm progress bar.
    """
    metrics = parse_stage_timing_logs(content)

    # If full pipeline latency was found from logs, return
    if "latency_ms_per_query" in metrics:
        return metrics

    # Fallback to tqdm progress bar matching
    matches = TQDM_REGEX.findall(content)
    if not matches:
        return metrics

    # Find final completed match (cur == total and total > 0), else take last match
    final_match = None
    for cur, total, elapsed, rate in reversed(matches):
        if cur == total and int(total) > 0:
            final_match = (cur, total, elapsed, rate)
            break
    if final_match is None:
        final_match = matches[-1]

    _, total, elapsed, rate = final_match
    total_queries = int(total)
    elapsed_sec = parse_tqdm_time_seconds(elapsed)

    parsed_ms = parse_tqdm_rate_to_ms(rate)
    if parsed_ms is not None:
        metrics["latency_ms_per_query"] = parsed_ms
    elif total_queries > 0 and elapsed_sec > 0:
        metrics["latency_ms_per_query"] = (elapsed_sec * 1000.0) / total_queries

    return metrics


def extract_latency_from_log_file(log_file: Path) -> Dict[str, float]:
    """Read a log file and extract latency metrics."""
    if not log_file.exists():
        logger.debug("Log file not found: %s", log_file)
        return {}

    try:
        content = log_file.read_bytes().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Could not read %s: %s", log_file, e)
        return {}

    return extract_latency_from_log_content(content)


def sync_latency_metrics(tests) -> None:
    """Finds log files for each evaluation in tests, extracts latency metrics,
    and updates aggregated.txt in place before to_dataframe().
    """
    if not hasattr(tests, "collection"):
        raise AttributeError(
            f"Provided tests object ({type(tests)}) has no 'collection' attribute"
        )

    for dataset_name, evaluations in tests.collection.items():
        eval_list = getattr(evaluations, "results", [])
        if not eval_list and hasattr(evaluations, "per_tags"):
            eval_list = list(evaluations.per_tags.values())

        for evaluation in eval_list:
            results_path = getattr(evaluation, "results", None)
            if results_path is None:
                continue
            results_path = Path(results_path)
            if not results_path.exists():
                logger.warning(
                    "Aggregated results file %s not found for dataset %s",
                    results_path,
                    dataset_name,
                )
                continue

            task_dir = results_path.parent

            # Look for candidate log files in order of priority
            candidate_files: List[Path] = []
            for name in [
                "evaluate.err",
                "evaluate.out",
                "job.err",
                "job.out",
                "task.err",
                "task.out",
            ]:
                p = task_dir / name
                if p.exists() and p not in candidate_files:
                    candidate_files.append(p)

            # Also check any other *.err, *.log, *.out files in task_dir
            for pattern in ["*.err", "*.log", "*.out"]:
                for p in sorted(task_dir.glob(pattern)):
                    if p not in candidate_files:
                        candidate_files.append(p)

            metrics_to_write: Dict[str, float] = {}
            for log_file in candidate_files:
                parsed = extract_latency_from_log_file(log_file)
                if parsed:
                    metrics_to_write.update(parsed)
                    if "latency_ms_per_query" in metrics_to_write:
                        break

            if not metrics_to_write:
                logger.debug(
                    "No latency metrics could be extracted for %s in %s",
                    dataset_name,
                    task_dir,
                )
                continue

            # Update aggregated.txt in-place (idempotent, preserving existing metrics)
            existing_text = results_path.read_text()
            existing_lines = existing_text.splitlines()
            updated_lines = []
            seen_metrics = set()

            for line in existing_lines:
                tokens = line.strip().split()
                if tokens and tokens[0] in metrics_to_write:
                    metric = tokens[0]
                    val = metrics_to_write[metric]
                    updated_lines.append(f"{metric:40s} {'all':10s} {val:.4f}")
                    seen_metrics.add(metric)
                elif line.strip():
                    updated_lines.append(line)

            for metric, val in metrics_to_write.items():
                if metric not in seen_metrics:
                    updated_lines.append(f"{metric:40s} {'all':10s} {val:.4f}")

            new_text = "\n".join(updated_lines) + "\n"
            if new_text != existing_text:
                results_path.write_text(new_text)
                logger.info(
                    "Updated latency metrics %s in %s",
                    metrics_to_write,
                    results_path,
                )
