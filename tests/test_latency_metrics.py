"""Unit and integration tests for query latency extraction and syncing.
Run with: `uv run pytest tests/test_latency_metrics.py -v`
"""

import logging
from pathlib import Path
import pytest

from latency_utils import (
    parse_tqdm_time_seconds,
    parse_tqdm_rate_to_ms,
    parse_stage_timing_logs,
    extract_latency_from_log_content,
    extract_latency_from_log_file,
    sync_latency_metrics,
)
from configuration import CE_FineTuning
from Evaluation.experiment import BaselinesConfig

logging.basicConfig(level=logging.INFO)


def test_parse_tqdm_time_seconds():
    assert parse_tqdm_time_seconds("26") == 26.0
    assert parse_tqdm_time_seconds("00:26") == 26.0
    assert parse_tqdm_time_seconds("03:25") == 205.0
    assert parse_tqdm_time_seconds("01:02:03") == 3723.0


def test_parse_tqdm_rate_to_ms():
    # queries/s
    ms_51 = parse_tqdm_rate_to_ms("51.68queries/s")
    assert ms_51 is not None
    assert pytest.approx(ms_51, 1e-3) == (1000.0 / 51.68)

    # query/s
    ms_4 = parse_tqdm_rate_to_ms("4.35query/s")
    assert ms_4 is not None
    assert pytest.approx(ms_4, 1e-3) == (1000.0 / 4.35)

    # s/query
    ms_sec = parse_tqdm_rate_to_ms("3.37s/query")
    assert ms_sec is not None
    assert pytest.approx(ms_sec, 1e-3) == 3370.0

    # ms/query
    ms_val = parse_tqdm_rate_to_ms("15.2ms/query")
    assert ms_val is not None
    assert pytest.approx(ms_val, 1e-3) == 15.2

    # it/s
    ms_it = parse_tqdm_rate_to_ms("10.0it/s")
    assert ms_it is not None
    assert pytest.approx(ms_it, 1e-3) == 100.0

    # s/it
    ms_sit = parse_tqdm_rate_to_ms("0.5s/it")
    assert ms_sit is not None
    assert pytest.approx(ms_sit, 1e-3) == 500.0

    # ms/it
    ms_msit = parse_tqdm_rate_to_ms("2.5ms/it")
    assert ms_msit is not None
    assert pytest.approx(ms_msit, 1e-3) == 2.5

    # Invalid string
    assert parse_tqdm_rate_to_ms("unknown_token") is None


def test_parse_stage_timing_logs():
    log_text = """
    INFO [xpmir.rankers.scorer] Stage 1 retrieval total time: 10.50s (7.65 ms/query)
    INFO [xpmir.rankers.scorer] Stage 2 reranking total time: 25.20s (18.35 ms/query)
    INFO [xpmir.rankers.scorer] Full two-stage pipeline completed in 35.70s (26.00 ms/query)
    """
    metrics = parse_stage_timing_logs(log_text)
    assert metrics["latency_ms_per_query"] == 26.0
    assert metrics["stage1_ms_per_query"] == 7.65
    assert metrics["stage2_ms_per_query"] == 18.35


def test_extract_latency_from_log_content_structured_priority():
    log_text = """
    1373/1373 [00:26<00:00, 51.68queries/s]
    Stage 1 retrieval total time: 10.50s (7.65 ms/query)
    Stage 2 reranking total time: 25.20s (18.35 ms/query)
    Full two-stage pipeline completed in 35.70s (26.00 ms/query)
    """
    metrics = extract_latency_from_log_content(log_text)
    # Structured pipeline log takes priority over tqdm rate
    assert metrics["latency_ms_per_query"] == 26.0
    assert metrics["stage1_ms_per_query"] == 7.65
    assert metrics["stage2_ms_per_query"] == 18.35


def test_extract_latency_from_log_content_tqdm_fallback():
    # Multiple tqdm lines in log, should pick the final completed bar
    log_text = """
    Evaluating queries:
    100/1373 [00:02<00:24, 50.00queries/s]
    500/1373 [00:10<00:16, 52.00queries/s]
    1373/1373 [00:26<00:00, 51.68queries/s]
    Completed retrieval.
    """
    metrics = extract_latency_from_log_content(log_text)
    assert "latency_ms_per_query" in metrics
    assert pytest.approx(metrics["latency_ms_per_query"], 1e-3) == (1000.0 / 51.68)
    assert "stage1_ms_per_query" not in metrics


def test_extract_latency_from_log_file(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.log"
    assert extract_latency_from_log_file(non_existent) == {}

    log_file = tmp_path / "task.log"
    log_file.write_text("349/349 [01:20<00:00, 4.35query/s]")
    metrics = extract_latency_from_log_file(log_file)
    assert pytest.approx(metrics["latency_ms_per_query"], 1e-3) == (1000.0 / 4.35)


def test_sync_latency_metrics_basic(tmp_path: Path):
    task_dir = tmp_path / "eval_task"
    task_dir.mkdir(parents=True)

    results_file = task_dir / "aggregated.txt"
    results_file.write_text(
        "nDCG@10                   all        0.4500\n"
        "RR@10                     all        0.3500\n"
    )

    err_file = task_dir / "evaluate.err"
    err_file.write_text(
        "Stage 1 retrieval total time: 5.00s (5.00 ms/query)\n"
        "Stage 2 reranking total time: 10.00s (10.00 ms/query)\n"
        "Full two-stage pipeline completed in 15.00s (15.00 ms/query)\n"
    )

    class MockEvaluation:
        def __init__(self, results):
            self.results = results

    class MockEvaluations:
        def __init__(self, eval_obj):
            self.results = [eval_obj]

    class MockCollection:
        def __init__(self, d):
            self.collection = d

    mock_tests = MockCollection({"msmarco": MockEvaluations(MockEvaluation(results_file))})

    # Run sync
    sync_latency_metrics(mock_tests)

    content = results_file.read_text()
    assert "latency_ms_per_query" in content
    assert "15.0000" in content
    assert "stage1_ms_per_query" in content
    assert "5.0000" in content
    assert "stage2_ms_per_query" in content
    assert "10.0000" in content
    assert "nDCG@10" in content
    assert "RR@10" in content

    # Idempotence: re-running should not duplicate lines
    sync_latency_metrics(mock_tests)
    lines = [line.strip() for line in results_file.read_text().strip().split("\n")]
    assert len(lines) == 5  # nDCG@10, RR@10, latency_ms_per_query, stage1_ms_per_query, stage2_ms_per_query


def test_config_get_latency_defaults():
    cfg = CE_FineTuning()
    assert hasattr(cfg, "get_latency")
    assert cfg.get_latency is False

    base_cfg = BaselinesConfig()
    assert hasattr(base_cfg, "get_latency")
    assert base_cfg.get_latency is False
