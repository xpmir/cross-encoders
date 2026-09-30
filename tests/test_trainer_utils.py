import logging
from unittest.mock import MagicMock
import pytest

from xpmir.letor.trainers.utils import wrap_collate_with_tokenizer


class DummyModelWithoutTokenizer:
    def __call__(self, records, **kwargs):
        return "called_without_tokenized"


class DummyModelWithNoneTokenizer:
    def get_tokenizer_fn(self):
        return None

    def __call__(self, records, **kwargs):
        return "called"


class DummyModelWithNonCallableTokenizer:
    def get_tokenizer_fn(self):
        return "not_a_callable"

    def __call__(self, records, **kwargs):
        return "called"


class DummyModelWithValidTokenizer:
    def get_tokenizer_fn(self):
        return lambda records: {"input_ids": [1, 2, 3]}

    def __call__(self, records, **kwargs):
        if "tokenized" in kwargs:
            return f"called_with_tokenized:{kwargs['tokenized']}"
        return "called_without_tokenized"


class DummyWrappedModel:
    def __init__(self, inner):
        self.module = inner


def test_wrap_collate_without_tokenizer_fn(caplog):
    model = DummyModelWithoutTokenizer()
    base_collate = lambda samples: {"records": samples}

    with caplog.at_level(logging.WARNING):
        collate_fn = wrap_collate_with_tokenizer(model, base_collate)

    assert "does not implement `get_tokenizer_fn()`" in caplog.text
    # Should return original collate function
    assert collate_fn is base_collate
    assert collate_fn(["item1", "item2"]) == {"records": ["item1", "item2"]}


def test_wrap_collate_with_none_tokenizer_fn(caplog):
    model = DummyModelWithNoneTokenizer()
    base_collate = lambda samples: {"records": samples}

    with caplog.at_level(logging.WARNING):
        collate_fn = wrap_collate_with_tokenizer(model, base_collate)

    assert "failed to grab a callable preprocessing function (got None)" in caplog.text
    assert collate_fn is base_collate
    assert collate_fn(["item1", "item2"]) == {"records": ["item1", "item2"]}


def test_wrap_collate_with_non_callable_tokenizer_fn(caplog):
    model = DummyModelWithNonCallableTokenizer()
    base_collate = lambda samples: {"records": samples}

    with caplog.at_level(logging.WARNING):
        collate_fn = wrap_collate_with_tokenizer(model, base_collate)

    assert "failed to grab a callable preprocessing function" in caplog.text
    assert collate_fn is base_collate


def test_wrap_collate_with_valid_tokenizer():
    model = DummyModelWithValidTokenizer()
    base_collate = lambda samples: {"records": samples}

    collate_fn = wrap_collate_with_tokenizer(model, base_collate)
    assert collate_fn is not base_collate

    batch = collate_fn(["doc1", "doc2"])
    assert batch["records"] == ["doc1", "doc2"]
    assert batch["tokenized_records"] == {"input_ids": [1, 2, 3]}


def test_wrap_collate_with_wrapped_model():
    inner = DummyModelWithValidTokenizer()
    wrapped = DummyWrappedModel(inner)
    base_collate = lambda samples: {"records": samples}

    collate_fn = wrap_collate_with_tokenizer(wrapped, base_collate)
    batch = collate_fn(["doc1", "doc2"])
    assert batch["records"] == ["doc1", "doc2"]
    assert batch["tokenized_records"] == {"input_ids": [1, 2, 3]}


def test_export_model_skips_when_torch_mocked(tmp_path, caplog, monkeypatch):
    import pandas as pd
    from experimaestro.experiments.mockmodule import FakeModule
    from training_utils import export_model
    import training_utils as tu

    dummy_torch = FakeModule("torch", None)
    monkeypatch.setattr(tu, "torch", dummy_torch)

    best_tags = {"base": "dummy_base", "from_experiment": "exp1"}
    csv_results = pd.DataFrame([{"dataset": "test", "ndcg": 0.5}])
    md_results = pd.DataFrame([{"dataset": "test", "ndcg": 0.5}])

    class DummyTask:
        class __xpm__:
            @staticmethod
            def tags():
                return best_tags

        def export_action(self, *args, **kwargs):
            raise RuntimeError("Should not be called when torch is mocked!")

    dummy_weight_task = DummyTask()

    with caplog.at_level(logging.WARNING):
        export_model(
            best_tags=best_tags,
            model_name="test_model",
            csv_results=csv_results,
            md_results=md_results,
            learners=[],
            all_weights=[dummy_weight_task],
            best_cfg=None,
            resultspath=tmp_path,
            save_runs=False,
        )

    assert "MODEL EXPORT SKIPPED - MOCKS DETECTED" in caplog.text
    assert (tmp_path / "models" / "test_model" / "results.csv").exists()
