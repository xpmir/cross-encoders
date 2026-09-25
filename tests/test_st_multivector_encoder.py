"""Tests for STMultiVectorEncoder, STModuleLoader, and PLAID integration.

Run with:
    uv run pytest tests/test_st_multivector_encoder.py -v
"""

from pathlib import Path
import pytest
import torch

from experimaestro import ObjectStore
from experimaestro.taskglobals import Env
from experimaestro.xpmutils import DirectoryContext
from datamaestro_ir.data import IDTextRecord, SimpleTextItem
from xpmir.letor.records import PointwiseItems, ProductItems
from xpmir.neural.sentence_transformers import (
    STMultiVectorEncoder,
    STModuleLoader,
    st_multivector_scorer,
)
from xpmir.neural.colbert import PylateColBERT, pylate_colbert
from xpmir.index.plaid import PlaidIndex, PlaidIndexBuilder, PlaidRetriever
from xpmir.test.utils.utils import SampleDocumentStore


BASE_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_st_multivector_encoder_forward_and_parity():
    """Verify forward pass on PointwiseItems & ProductItems and parity with raw MultiVectorEncoder."""
    from sentence_transformers import MultiVectorEncoder as RawMultiVectorEncoder

    scorer = st_multivector_scorer(model_id=BASE_MODEL_ID).instance()
    scorer.initialize()
    scorer.eval()

    raw_model = scorer.st_model

    queries = ["What is the capital of France?", "Who wrote Hamlet?"]
    documents = [
        "Paris is the capital of France.",
        "William Shakespeare wrote Hamlet.",
    ]

    # 1. Test PointwiseItems (paired scoring)
    pointwise = PointwiseItems.from_texts(topics=queries, documents=documents)
    with torch.no_grad():
        out_pointwise = scorer(pointwise)

    assert isinstance(out_pointwise, torch.Tensor)
    assert out_pointwise.shape == (2,)

    # Raw parity check for paired queries & documents
    with torch.no_grad():
        raw_q = raw_model.encode_query(queries, convert_to_numpy=False, normalize_embeddings=True)
        raw_d = raw_model.encode_document(documents, convert_to_numpy=False, normalize_embeddings=True)
        expected_paired = raw_model.similarity_pairwise(raw_q, raw_d)

    assert torch.allclose(out_pointwise.cpu(), expected_paired.cpu(), atol=1e-4)

    # 2. Test ProductItems (all pairs query x doc)
    product = ProductItems()
    product.add_topics(*[{"text_item": SimpleTextItem(q)} for q in queries])
    product.add_documents(*[{"text_item": SimpleTextItem(d)} for d in documents])
    with torch.no_grad():
        out_product = scorer(product)

    assert isinstance(out_product, torch.Tensor)
    assert out_product.shape == (4,)  # 2 queries * 2 docs flattened

    with torch.no_grad():
        expected_matrix = raw_model.similarity(raw_q, raw_d).flatten()

    assert torch.allclose(out_product.cpu(), expected_matrix.cpu(), atol=1e-4)


def test_st_multivector_encoder_plaid_embeddings():
    """Verify document_token_embeddings and query_token_embeddings protocols."""
    scorer = st_multivector_scorer(model_id=BASE_MODEL_ID).instance()
    scorer.initialize()
    scorer.eval()

    docs = [
        IDTextRecord(id="doc1", text_item="Paris is the capital of France."),
        IDTextRecord(id="doc2", text_item="Berlin is the capital of Germany."),
    ]

    # Document token embeddings
    doc_embs = scorer.document_token_embeddings(docs)
    assert isinstance(doc_embs, list)
    assert len(doc_embs) == 2
    for emb in doc_embs:
        assert isinstance(emb, torch.Tensor)
        assert emb.ndim == 2
        assert emb.shape[1] == scorer.dimension

    # Query token embeddings
    queries = [
        IDTextRecord(id="q1", text_item="capital of France"),
        IDTextRecord(id="q2", text_item="Germany"),
    ]
    query_tensor = scorer.query_token_embeddings(queries)
    assert isinstance(query_tensor, torch.Tensor)
    assert query_tensor.ndim == 3
    assert query_tensor.shape[0] == 2
    assert query_tensor.shape[2] == scorer.dimension


def test_st_multivector_encoder_plaid_builder_and_retriever(tmp_path: Path):
    """Verify end-to-end integration with PlaidIndexBuilder and PlaidRetriever."""
    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir(exist_ok=True)
    context = DirectoryContext(tmp_path)
    objects = ObjectStore()

    doc_store = SampleDocumentStore.C(num_docs=4)
    scorer_cfg = st_multivector_scorer(model_id=BASE_MODEL_ID)

    builder = PlaidIndexBuilder.C(
        documents=doc_store,
        encoder=scorer_cfg,
        buffer_size=2,
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
        topk=2,
    ).instance()
    retriever.initialize()

    query_record = IDTextRecord(id="q1", text_item="sample query text")
    results = retriever.retrieve(query_record)
    assert len(results) <= 2
    for scored_doc in results:
        assert scored_doc.score is not None
        assert scored_doc.document is not None


def test_st_module_loader_multivector(tmp_path: Path):
    """Verify that STModuleLoader properly saves and loads STMultiVectorEncoder checkpoints."""
    scorer = st_multivector_scorer(model_id=BASE_MODEL_ID).instance()
    scorer.initialize()

    ckpt_dir = tmp_path / "checkpoint"
    scorer.save_model(ckpt_dir)
    assert (ckpt_dir / "modules.json").is_file()

    # Load via STModuleLoader
    eval_cfg = STMultiVectorEncoder.C(model_id="invalid/model/path")
    loader_cfg = eval_cfg.loader_config(ckpt_dir)
    loader = loader_cfg.instance()

    assert isinstance(loader, STModuleLoader)
    loader.execute()

    loaded_scorer = loader.value
    assert loaded_scorer._initialized is True
    assert loaded_scorer.model_id == str(ckpt_dir)
    assert loaded_scorer.st_model is not None


def test_pylate_colbert_deprecation():
    """Verify that PylateColBERT issues a deprecation warning and delegates to STMultiVectorEncoder."""
    with pytest.deprecated_call():
        pylate_scorer = PylateColBERT.C(model_id=BASE_MODEL_ID).instance()

    with pytest.deprecated_call():
        factory_scorer, _ = pylate_colbert(model_id=BASE_MODEL_ID)

    assert isinstance(pylate_scorer, STMultiVectorEncoder)
    pylate_scorer.initialize()
    assert pylate_scorer._initialized is True


def test_st_multivector_encoder_forward_list_inputs_and_backward():
    """Verify forward pass accepts a plain list of PointwiseItem (from DataLoader) and supports autograd backprop."""
    from xpmir.letor.records import PointwiseItem
    from datamaestro_ir.data import SimpleTextItem

    scorer = st_multivector_scorer(model_id=BASE_MODEL_ID).instance()
    scorer.initialize()
    scorer.train()

    items = [
        PointwiseItem(
            topic={"text_item": SimpleTextItem("What is the capital of France?")},
            document={"text_item": SimpleTextItem("Paris is the capital of France.")},
            relevance=1.0,
        ),
        PointwiseItem(
            topic={"text_item": SimpleTextItem("Who wrote Hamlet?")},
            document={"text_item": SimpleTextItem("William Shakespeare wrote Hamlet.")},
            relevance=1.0,
        ),
    ]

    scores = scorer(items)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)
    assert scores.requires_grad is True

    target = torch.tensor([1.0, 1.0], dtype=torch.float32, device=scores.device)
    loss = torch.nn.MSELoss()(scores, target)
    loss.backward()

    grads = [p.grad for p in scorer.st_model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed on model parameters during backward pass"


def test_st_multivector_encoder_tokenizer_methods():
    """Verify tokenizer methods on STMultiVectorEncoder."""
    scorer = st_multivector_scorer(model_id=BASE_MODEL_ID).instance()
    scorer.initialize()

    tokenizer = scorer.get_tokenizer()
    assert tokenizer is not None

    vocab_size = scorer.vocabulary_size()
    assert isinstance(vocab_size, int)
    assert vocab_size > 0

    tok_id = scorer.tok2id("paris")
    assert isinstance(tok_id, int)
    tok = scorer.id2tok(tok_id)
    assert tok == "paris"
