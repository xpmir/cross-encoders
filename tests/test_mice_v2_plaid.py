"""
Tests for ModernBERT MICE V2 PLAID Indexation & Retrieval.
Run with: `uv run pytest tests/test_mice_v2_plaid.py -v`
"""

import tempfile
import torch
import pytest
from datamaestro_ir.data.base import IDTextRecord
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, MiceV2DocumentEncoder
from MICE.modeling.mice import MICETokenizedTexts, QueryDocInput
from xpmir.letor.records import PointwiseItems

import logging
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_document_encoder(model_id):
    """Test get_document_encoder and document token embedding generation."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    docs = [
        IDTextRecord(id="doc1", text="Paris is the capital of France."),
        IDTextRecord(id="doc2", text="Berlin is the capital of Germany."),
    ]

    with torch.no_grad():
        token_embs = doc_encoder.document_token_embeddings(docs)

    assert len(token_embs) == 2, f"Expected 2 document embeddings, got {len(token_embs)}"
    assert isinstance(token_embs[0], torch.Tensor), "Embedding should be a Tensor"
    assert token_embs[0].ndim == 2, f"Expected shape [seq_len, dim], got {token_embs[0].shape}"
    logging.info(f"Document 1 token embeddings shape: {token_embs[0].shape}")


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_doc_hidden_states_forward(model_id):
    """Test forward pass with decompressed document hidden states (PLAID pipeline)."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    queries = ["What is the capital of France?", "Where is Berlin?"]
    documents = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    # 1. Standard forward pass
    with torch.no_grad():
        standard_scores = scorer(input_records)

    # 2. Forward pass with pre-encoded doc_hidden_states using encode_documents
    doc_records = [IDTextRecord(id=f"doc_{i}", text=text) for i, text in enumerate(documents)]
    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    with torch.no_grad():
        doc_hidden_states = doc_encoder.document_token_embeddings(doc_records)
        tokenized_q = scorer.batch_tokenize(input_records, input_nature=QueryDocInput.QUERY)
        plaid_scores = scorer(
            tokenized=MICETokenizedTexts(tokenized_q=tokenized_q, tokenized_docs=None),
            doc_hidden_states=doc_hidden_states,
        )

    logging.info(f"Standard scores: {standard_scores}")
    logging.info(f"PLAID doc_hidden scores: {plaid_scores}")

    assert torch.allclose(standard_scores, plaid_scores, atol=1e-4), \
        f"Mismatch between standard forward ({standard_scores}) and PLAID doc_hidden forward ({plaid_scores})"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_normalize_doc_embeds(model_id):
    """Test document state L2 normalization when normalize_doc_embeds=True."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        normalize_doc_embeds=True,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    docs = [
        IDTextRecord(id="doc1", text="Paris is the capital of France."),
        IDTextRecord(id="doc2", text="Berlin is the capital of Germany."),
    ]
    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    with torch.no_grad():
        token_embs = doc_encoder.document_token_embeddings(docs)

    for i, emb in enumerate(token_embs):
        norms = torch.norm(emb, p=2, dim=-1)
        logging.info(f"Doc {i} token norms (normalize_doc_embeds=True): {norms}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Expected unit L2 norm (1.0), got {norms}"

    # Verify parity between standard forward pass and pre-encoded doc_hidden_states
    queries = ["What is the capital of France?", "Where is Berlin?"]
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    with torch.no_grad():
        standard_scores = scorer(input_records)
        tokenized_q = scorer.batch_tokenize(input_records, input_nature=QueryDocInput.QUERY)
        plaid_scores = scorer(
            tokenized=MICETokenizedTexts(tokenized_q=tokenized_q, tokenized_docs=None),
            doc_hidden_states=token_embs,
        )

    assert torch.allclose(standard_scores, plaid_scores, atol=1e-4), \
        f"Score mismatch with normalize_doc_embeds=True: {standard_scores} vs {plaid_scores}"
