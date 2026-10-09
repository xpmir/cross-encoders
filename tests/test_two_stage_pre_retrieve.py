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
    # Set chunk size to 2 to test micro-batch streaming with multiple chunks
    retriever.batchsize = 2

    two_stage_cfg = TwoStageRetriever.C(
        retriever=retriever_cfg,
        scorer=scorer,
        batchsize=4,
        top_k=2,
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
    for qid in queries:
        assert len(results[qid]) == 2
        assert isinstance(results[qid][0], ScoredDocument)
