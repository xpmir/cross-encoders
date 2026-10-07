"""
Unit tests for index size measurement and experimaestro dependency graph traversal.
Run with: `uv run pytest tests/test_index_size.py -v`
"""

import tempfile
from pathlib import Path
import pytest
from experimaestro import experiment, RunMode, Task, Meta, Param, Config, field, PathGenerator
from xpmir.index.plaid import PlaidIndexBuilder
from xpmir.index.sparse import SparseRetrieverIndexBuilder, AbstractSparseRetrieverIndexBuilder

from index_utils import (
    get_path_size_mb,
    find_index_builders,
    sync_index_size_metrics,
)


def test_get_path_size_mb():
    """Verify get_path_size_mb returns accurate sizes and handles non-existent paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        index_dir = tmp_path / "index_dir"
        index_dir.mkdir()

        # Create a 2MB file
        data_file = index_dir / "data.bin"
        data_file.write_bytes(b"0" * (2 * 1024 * 1024))

        size_mb = get_path_size_mb(index_dir)
        # Should be approximately 2.0 MB
        assert 1.9 <= size_mb <= 2.2

        # Non-existent path raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_path_size_mb(tmp_path / "non_existent_folder")


class MockIndex(Config):
    index_path: Param[Path]


class MockPlaidIndexBuilder(Task):
    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))

    def task_outputs(self, dep):
        return dep(MockIndex.C(index_path=self.index_path))

    def execute(self):
        pass


class MockSparseRetrieverIndexBuilder(Task):
    index_path: Meta[Path] = field(default_factory=PathGenerator("sparse-index"))

    def task_outputs(self, dep):
        return dep(MockIndex.C(index_path=self.index_path))

    def execute(self):
        pass


class MockRetriever(Config):
    plaid_index: Param[MockIndex]
    sparse_index: Param[MockIndex]


class MockEvaluate(Task):
    retriever: Param[MockRetriever]
    aggregated: Meta[Path] = field(default_factory=PathGenerator("aggregated.txt"))

    def execute(self):
        pass


def test_find_index_builders_graph_traversal():
    """Verify that find_index_builders traverses experimaestro dependency graph and finds all builders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with experiment(Path(tmpdir), "test_exp", run_mode=RunMode.DRY_RUN) as xp:
            plaid_idx = MockPlaidIndexBuilder.C().submit()
            sparse_idx = MockSparseRetrieverIndexBuilder.C().submit()
            retriever = MockRetriever.C(plaid_index=plaid_idx, sparse_index=sparse_idx)
            eval_task = MockEvaluate.C(retriever=retriever).submit()

            builder_classes = (MockPlaidIndexBuilder, MockSparseRetrieverIndexBuilder)
            builders = find_index_builders(eval_task, builder_classes=builder_classes)
            assert len(builders) == 2
            assert any(isinstance(b, MockPlaidIndexBuilder) for b in builders)
            assert any(isinstance(b, MockSparseRetrieverIndexBuilder) for b in builders)


def test_find_index_builders_default_classes():
    """Verify that find_index_builders defaults to PlaidIndexBuilder and AbstractSparseRetrieverIndexBuilder."""
    eval_task = MockEvaluate.C()
    # When no matching builders exist in unsubmitted task, returns empty list without error
    builders = find_index_builders(eval_task)
    assert builders == []


class DummyEvaluation:
    def __init__(self, task, results_path: Path):
        self.__xpm__ = getattr(task, "__xpm__", None)
        self.results = results_path


class DummyEvaluations:
    def __init__(self, evals):
        self.results = evals
        self.per_tags = {("dummy_tag", "val"): e for e in evals}


class DummyCollection:
    def __init__(self, evals_dict):
        self.collection = evals_dict


def test_sync_index_size_metrics_idempotency_and_formatting():
    """Verify that sync_index_size_metrics writes <ClassName>_index_size to aggregated.txt and is idempotent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with experiment(Path(tmpdir), "test_exp", run_mode=RunMode.DRY_RUN) as xp:
            plaid_builder = MockPlaidIndexBuilder.C().submit()
            sparse_builder = MockSparseRetrieverIndexBuilder.C().submit()
            retriever = MockRetriever.C(plaid_index=plaid_builder, sparse_index=sparse_builder)
            eval_task = MockEvaluate.C(retriever=retriever).submit()

            # Create physical directories for the index outputs with dummy data
            plaid_path = Path(plaid_builder.index_path)
            plaid_path.mkdir(parents=True, exist_ok=True)
            (plaid_path / "plaid.bin").write_bytes(b"x" * (1024 * 1024))  # 1 MB

            sparse_path = Path(sparse_builder.index_path)
            sparse_path.mkdir(parents=True, exist_ok=True)
            (sparse_path / "sparse.bin").write_bytes(b"y" * (2 * 1024 * 1024))  # 2 MB

            # Setup aggregated.txt
            agg_path = Path(tmpdir) / "aggregated.txt"
            agg_path.write_text("RR@10                     all        0.7500\n")

            dummy_eval = DummyEvaluation(eval_task, agg_path)
            tests = DummyCollection({"test_ds": DummyEvaluations([dummy_eval])})

            builder_classes = (MockPlaidIndexBuilder, MockSparseRetrieverIndexBuilder)

            # First run: should append both metrics
            sync_index_size_metrics(tests, builder_classes=builder_classes)

            content = agg_path.read_text()
            assert "index_size_MockPlaidIndexBuilder" in content
            assert "index_size_MockSparseRetrieverIndexBuilder" in content
            lines = [l for l in content.splitlines() if "index_size" in l]
            assert len(lines) == 2

            # Verify trec-eval format: splits cleanly into 3 fields
            for line in lines:
                parts = line.split()
                assert len(parts) == 3
                assert parts[1] == "all"
                assert float(parts[2]) > 0.0

            # Second run: should be idempotent (no duplicate lines)
            sync_index_size_metrics(tests, builder_classes=builder_classes)
            content_after = agg_path.read_text()
            lines_after = [l for l in content_after.splitlines() if "index_size" in l]
            assert len(lines_after) == 2
            assert content == content_after

            # Verify xpmir's TrecAdhocResults parses both metrics into float values
            from xpmir.evaluation import TrecAdhocResults
            trec_results = TrecAdhocResults.C(id="test", results=agg_path, detailed=Path(tmpdir) / "detailed.dat")
            parsed = trec_results.get_results()
            assert "index_size_MockPlaidIndexBuilder" in parsed
            assert "index_size_MockSparseRetrieverIndexBuilder" in parsed
            assert float(parsed["index_size_MockPlaidIndexBuilder"]) == pytest.approx(1.0, 0.01)
            assert float(parsed["index_size_MockSparseRetrieverIndexBuilder"]) == pytest.approx(2.0, 0.01)


def test_process_experiment_results_syncs_index_size_when_plaid(tmp_path):
    """Verify that process_experiment_results automatically triggers sync_index_size_metrics when plaid is enabled."""
    from unittest.mock import MagicMock, patch
    import pandas as pd
    from training_utils import process_experiment_results
    from configuration import CE_FineTuning, PlaidConfiguration

    columns = pd.MultiIndex.from_tuples([
        ("dataset", ""),
        ("tag", "base"),
        ("tag", "plaid_retriever"),
        ("metric", "RR@10"),
    ])
    eval_df = pd.DataFrame(
        [["BEIR13", "colbert-test", True, 0.75]],
        columns=columns,
    )

    mock_tests = MagicMock()
    mock_tests.to_dataframe.return_value = eval_df
    mock_tests.collection = {}

    mock_helper = MagicMock()
    mock_helper.xp.resultspath = tmp_path

    cfg = CE_FineTuning(use_plaid=True)
    all_tags = [{"base": "colbert-test"}]
    config_map = {frozenset([("base", "colbert-test")]): cfg}

    with patch("index_utils.sync_index_size_metrics") as mock_sync:
        process_experiment_results(
            tests=mock_tests,
            all_tags=all_tags,
            config_map=config_map,
            cfg=cfg,
            helper=mock_helper,
            learners=[],
            all_weights=[],
            get_name_fn=lambda tags: "ColBERT-Test",
        )
        assert mock_sync.call_count == 1
        assert mock_sync.call_args[0][0] is mock_tests
