"""Unit tests for PlaidRetriever and the multi-vector PLAID indexing pipeline."""

from pathlib import Path
import torch
import pytest

from experimaestro import ObjectStore, experiment
from experimaestro.taskglobals import Env
from experimaestro.xpmutils import DirectoryContext
from datamaestro_ir.data import IDTextRecord
from xpmir.test.utils.utils import SampleDocumentStore
from xpmir.index.plaid import PlaidIndex, PlaidIndexBuilder, PlaidRetriever
from xpmir.neural.sentence_transformers import st_multivector_scorer
from retrievers import plaid_retriever
from index_utils import _indexes
from configuration import PlaidConfiguration
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer


def test_modernbert_mice_multivector_protocol():
    """Verify ModernBertMiceScorer implements the multi-vector protocol for PLAID."""
    model_id = "jhu-clsp/ettin-encoder-17m"
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    # Check duck-typed protocol properties
    assert hasattr(scorer, "dimension")
    assert isinstance(scorer.dimension, int)
    assert scorer.dimension > 0
    assert hasattr(scorer, "model")
    assert isinstance(scorer.model, torch.nn.Module)

    # Check document_token_embeddings
    docs = [
        IDTextRecord(id="d1", text="A sunny day in Paris."),
        IDTextRecord(id="d2", text="Deep learning for retrieval."),
    ]
    with torch.no_grad():
        doc_embs = scorer.document_token_embeddings(docs)
    assert isinstance(doc_embs, list)
    assert len(doc_embs) == 2
    for t in doc_embs:
        assert isinstance(t, torch.Tensor)
        assert t.ndim == 2
        assert t.shape[1] == scorer.dimension

    # Check query_token_embeddings
    queries = [
        IDTextRecord(id="q1", text="Paris weather"),
        IDTextRecord(id="q2", text="neural search"),
    ]
    with torch.no_grad():
        q_embs = scorer.query_token_embeddings(queries)
    assert isinstance(q_embs, torch.Tensor)
    assert q_embs.ndim == 3
    assert q_embs.shape[0] == 2
    assert q_embs.shape[2] == scorer.dimension


def test_plaid_retriever_retrieve_all_batching(tmp_path: Path):
    """Test that PlaidRetriever.retrieve_all produces identical results to sequential retrieve."""
    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir(exist_ok=True)
    context = DirectoryContext(tmp_path)
    objects = ObjectStore()

    doc_store = SampleDocumentStore.C(num_docs=8)
    scorer_cfg = st_multivector_scorer(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        doc_maxlen=32,
        query_maxlen=16,
    )

    builder = PlaidIndexBuilder.C(
        documents=doc_store,
        encoder=scorer_cfg,
        batch_size=4,
        buffer_size=4,
        n_bits=2,
        kmeans_niters=2,
        compress_only=False,
    )
    builder_instance = builder.instance(context=context, objects=objects)
    builder_instance.execute()

    index = PlaidIndex.C(
        documents=doc_store,
        index_path=builder_instance.index_path,
        compress_only=False,
    )

    retriever = PlaidRetriever.C(
        encoder=scorer_cfg,
        index=index,
        topk=3,
        batch_size=2,  # test batching with size 2 across 4 queries
    ).instance()
    retriever.initialize()

    queries = {
        f"q_{i}": IDTextRecord(id=f"q_{i}", text_item=f"Query {i} information retrieval")
        for i in range(4)
    }

    # Sequential retrieve
    sequential_results = {
        qid: retriever.retrieve(q_record)
        for qid, q_record in queries.items()
    }

    # Batched retrieve_all
    batched_results = retriever.retrieve_all(queries)

    assert set(sequential_results.keys()) == set(batched_results.keys())
    for qid in queries:
        seq_docs = [sd.document["id"] for sd in sequential_results[qid]]
        batch_docs = [sd.document["id"] for sd in batched_results[qid]]
        assert seq_docs == batch_docs, f"Mismatch for query {qid}: {seq_docs} vs {batch_docs}"


def test_plaid_retriever_factory(tmp_path: Path):
    """Test that plaid_retriever factory correctly instantiates PlaidRetriever within an experiment."""
    _indexes.clear()
    with experiment(tmp_path, "test_exp"):
        doc_store = SampleDocumentStore.C(num_docs=4)
        scorer_cfg = st_multivector_scorer(
            model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            doc_maxlen=32,
            query_maxlen=16,
        )

        class MockConfig:
            plaid = PlaidConfiguration(n_ivf_probe=4, n_bits=2)
            retrieval = type("MockRetrieval", (), {"k": 100})()

        retriever_c = plaid_retriever(
            cfg=MockConfig(),
            name="test_plaid",
            encoder=scorer_cfg,
            documents=doc_store,
            launcher_index=None,
            topk=5,
        )

        assert isinstance(retriever_c, PlaidRetriever.C)
        assert retriever_c.topk == 5
        assert retriever_c.n_ivf_probe == 4


def test_retrievers_only_aggregations():
    """Verify that dataframe aggregation works when retrievers_only is True (no scorer tag)."""
    import pandas as pd
    from training_utils import add_dataset_aggregations
    from format import aggregations

    df = pd.DataFrame(
        {
            "dataset": ["trec2019", "nfcorpus", "trec2019", "nfcorpus"],
            "first_stage": ["plaid", "plaid", "bm25", "bm25"],
            "R@100": [0.65, 0.45, 0.55, 0.35],
            "R@1000": [0.85, 0.70, 0.75, 0.60],
        }
    )

    tag_names = ["first_stage"]
    group_by_tags = [col for col in sorted(list(tag_names)) if col in df.columns]

    agg_df = add_dataset_aggregations(
        df,
        group_by_cols=group_by_tags,
        aggregations=aggregations,
        add_mean=True,
    )
    assert not agg_df.empty
    assert "mean" in agg_df["dataset"].values
