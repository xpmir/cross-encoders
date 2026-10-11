import pytest
from unittest.mock import MagicMock
from datamaestro_ir.data.base import IDTextRecord, ScoredDocument
from xpmir.rankers.scorer import (
    TwoStageRetriever,
    ReRankingDataset,
    AbstractModuleScorer,
)
import torch
import torch.nn as nn


from xpmir.rankers.retriever import Retriever


class DummyScorer(AbstractModuleScorer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, batch_items, tokenized=None):
        return torch.tensor([float(i) for i in range(len(batch_items))])


class DummyRetriever(Retriever):
    def initialize(self):
        pass

    def retrieve(self, record: IDTextRecord):
        # Return 3 dummy scored documents per query
        return [
            ScoredDocument(
                IDTextRecord(id=f"doc_{record['id']}_{i}", text=f"content {i}"),
                float(10 - i),
            )
            for i in range(3)
        ]


def test_two_stage_retriever_retrieve_all_dataloader():
    scorer = DummyScorer.C()
    retriever = DummyRetriever.C()

    two_stage_cfg = TwoStageRetriever.C(
        retriever=retriever,
        scorer=scorer,
        batchsize=4,
        top_k=2,
    )
    two_stage = two_stage_cfg.instance()
    two_stage.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(5)
    }

    results = two_stage.retrieve_all(queries)

    assert len(results) == 5
    for qid in queries:
        assert len(results[qid]) == 2  # top_k == 2
        assert isinstance(results[qid][0], ScoredDocument)


def test_two_stage_retriever_micro_batch_streaming():
    scorer = DummyScorer.C()
    retriever_cfg = DummyRetriever.C()
    retriever = retriever_cfg.instance()

    two_stage_cfg = TwoStageRetriever.C(
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=4,
        top_k=2,
        chunk_size=2,
    )
    two_stage = two_stage_cfg.instance()
    two_stage.retriever = retriever
    two_stage.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(7)
    }

    results = two_stage.retrieve_all(queries)

    assert len(results) == 7
    assert len(two_stage.last_candidates) == 7
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)


def test_two_stage_retriever_all_in_one_go():
    scorer = DummyScorer.C()
    retriever_cfg = DummyRetriever.C()
    retriever = retriever_cfg.instance()

    # chunk_size=None (default): retrieves all candidates in one go
    two_stage_cfg = TwoStageRetriever.C(
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=4,
        top_k=2,
        chunk_size=None,
    )
    two_stage = two_stage_cfg.instance()
    two_stage.retriever = retriever
    two_stage.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(6)
    }

    results = two_stage.retrieve_all(queries)

    assert len(results) == 6
    assert len(two_stage.last_candidates) == 6
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)


from pathlib import Path
from MICE.plaid_mice import MicePlaidRetrieverv2, PlaidReRankingDataset
from xpmir.index.plaid import PlaidIndex
from xpmir.test.utils.utils import SampleDocumentStore


from xpmir.text.encoders import TokenizedTexts


class DummyMiceScorer(AbstractModuleScorer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def batch_tokenize(self, queries, options=None, input_nature=None):
        return TokenizedTexts(
            tokens=None,
            ids=torch.zeros(len(queries), 10, dtype=torch.long),
            lens=[10] * len(queries),
            mask=torch.ones(len(queries), 10, dtype=torch.long),
        )

    def forward(self, batch_items, tokenized=None, doc_hidden_states=None):
        assert doc_hidden_states is not None
        assert len(doc_hidden_states) == len(batch_items)
        return torch.tensor([float(i) for i in range(len(batch_items))])


class DummyPlaidIndex(PlaidIndex):
    def get_document_tokens(self, docids, device=""):
        return [torch.randn(5, 10) for _ in docids]


def test_mice_plaid_retriever_v2_retrieve_all():
    scorer = DummyMiceScorer.C()
    retriever_cfg = DummyRetriever.C()
    retriever = retriever_cfg.instance()
    index = DummyPlaidIndex.C(
        documents=SampleDocumentStore.C(num_docs=4), index_path=Path("/tmp/dummy")
    )

    mice_plaid_cfg = MicePlaidRetrieverv2.C(
        index=index,
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=4,
        top_k=2,
    )
    mice_plaid = mice_plaid_cfg.instance()
    mice_plaid.retriever = retriever
    mice_plaid.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(5)
    }

    results = mice_plaid.retrieve_all(queries)

    assert len(results) == 5
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)


def test_mice_plaid_retriever_v2_chunk_size():
    scorer = DummyMiceScorer.C()
    retriever_cfg = DummyRetriever.C()
    retriever = retriever_cfg.instance()
    index = DummyPlaidIndex.C(
        documents=SampleDocumentStore.C(num_docs=4), index_path=Path("/tmp/dummy")
    )

    mice_plaid_cfg = MicePlaidRetrieverv2.C(
        index=index,
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=4,
        top_k=2,
        chunk_size=2,
    )
    mice_plaid = mice_plaid_cfg.instance()
    mice_plaid.retriever = retriever
    mice_plaid.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(5)
    }

    results = mice_plaid.retrieve_all(queries)

    assert len(results) == 5
    assert len(mice_plaid.last_candidates) == 5
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)


def test_plaid_reranking_dataset_backwards_compatible():
    retriever = DummyRetriever.C().instance()
    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(3)
    }
    ds = PlaidReRankingDataset(queries, retriever)
    items = list(ds.iter_shard(0, 1))
    assert len(items) == 9  # 3 queries * 3 docs each


import pickle
from MICE.plaid_mice import MiceRerankingCollate
from xpmir.letor.records import PointwiseItem


def test_mice_reranking_collate_decompression():
    scorer = DummyMiceScorer.C()
    index = DummyPlaidIndex.C(
        documents=SampleDocumentStore.C(num_docs=4), index_path=Path("/tmp/dummy")
    )
    collate_with_index = MiceRerankingCollate(scorer, index=index, max_query_length=32)
    batch = [
        PointwiseItem(
            topic=IDTextRecord(id="q1", text="query 1"),
            document=IDTextRecord(id="d1", text="doc 1"),
        ),
        PointwiseItem(
            topic=IDTextRecord(id="q2", text="query 2"),
            document=IDTextRecord(id="d2", text="doc 2"),
        ),
    ]
    out = collate_with_index(batch)
    assert "doc_token_embeddings" in out
    assert len(out["doc_token_embeddings"]) == 2
    assert out["doc_token_embeddings"][0].shape == (5, 10)
    assert out["doc_ids"] == ["d1", "d2"]
    assert out["tokenized_queries"] is not None

    # Test with index=None
    collate_without_index = MiceRerankingCollate(scorer, index=None)
    out_no_idx = collate_without_index(batch)
    assert "doc_token_embeddings" not in out_no_idx
    assert out_no_idx["doc_ids"] == ["d1", "d2"]


def test_plaid_index_pickling_state():
    index = DummyPlaidIndex.C(
        documents=SampleDocumentStore.C(num_docs=4), index_path=Path("/tmp/dummy")
    )

    class MockUnpicklable:
        def __reduce__(self):
            raise TypeError("Cannot pickle this context")

    index._fast_plaid = MockUnpicklable()
    index._fast_plaid_device = "cuda:0"

    pickled = pickle.dumps(index)
    unpickled = pickle.loads(pickled)
    assert getattr(unpickled, "_fast_plaid", None) is None
    assert getattr(unpickled, "_fast_plaid_device", None) is None


def test_mice_plaid_retriever_v2_multiprocessing_workers(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    scorer = DummyMiceScorer.C()
    retriever_cfg = DummyRetriever.C()
    retriever = retriever_cfg.instance()
    index = DummyPlaidIndex.C(
        documents=SampleDocumentStore.C(num_docs=4), index_path=Path("/tmp/dummy")
    )

    mice_plaid_cfg = MicePlaidRetrieverv2.C(
        index=index,
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=2,
        top_k=2,
    )
    mice_plaid = mice_plaid_cfg.instance()
    mice_plaid.retriever = retriever
    mice_plaid.initialize()

    queries = {
        f"q{i}": IDTextRecord(id=f"q{i}", text=f"query text {i}")
        for i in range(4)
    }

    results = mice_plaid.retrieve_all(queries)

    assert len(results) == 4
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)
