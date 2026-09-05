"""
Parity test comparing HFCrossScorer and STCrossEncoder outputs.

Ensures that both HFCrossScorer (HuggingFace Transformers backbone) and
STCrossEncoder (SentenceTransformers library wrapper) produce consistent scores
when evaluated on identical query-document pairs.
"""

import pytest
import torch
import torch.nn as nn
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.neural.sentence_transformers import st_cross_scorer
from xpmir.letor.records import PointwiseItems


@pytest.mark.parametrize(
    "model_id",
    [
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "BAAI/bge-reranker-base",
    ],
)
def test_st_hf_cross_encoder_parity(model_id):
    """Verify output score consistency between HFCrossScorer and STCrossEncoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    queries = [
        "Which planet is known as the Red Planet?",
        "What is the capital of France?",
    ]

    documents = [
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Paris is the capital and most populous city of France.",
    ]

    records = PointwiseItems.from_texts(topics=queries, documents=documents)

    # 1. Setup HFCrossScorer (ensure init task executes on the instantiated encoder)
    hf_cfg, hf_tasks = hf_cross_scorer(hf_id=model_id)
    hf_scorer = hf_cfg.instance()
    hf_scorer.initialize()
    for task in hf_tasks:
        t = task.instance()
        t.model = hf_scorer.encoder
        t.execute()
    hf_scorer.to(device)
    hf_scorer.eval()

    # 2. Setup STCrossEncoder
    st_cfg = st_cross_scorer(model_id=model_id)
    st_scorer = st_cfg.instance()
    st_scorer.initialize()
    st_scorer.to(device)
    st_scorer.eval()

    # 3. Compute raw scores
    with torch.no_grad():
        hf_logits = hf_scorer(records).view(-1)
        st_scores = st_scorer(records).view(-1)

    # Apply ST activation function if STCrossEncoder uses one (e.g. Sigmoid)
    activation_fn = getattr(st_scorer.st_model, "activation_fn", None)
    if activation_fn is not None and not isinstance(activation_fn, nn.Identity):
        hf_scores = activation_fn(hf_logits)
    else:
        hf_scores = hf_logits

    # Print diagnostic comparison
    print(f"\n--- Parity Check for {model_id} ---")
    print(f"HFCrossScorer raw logits: {hf_logits}")
    print(f"HFCrossScorer scores (after activation): {hf_scores}")
    print(f"STCrossEncoder scores: {st_scores}")
    print(f"Absolute difference:  {(hf_scores - st_scores).abs()}")

    # Compare tokenization details if there is a mismatch
    if not torch.allclose(hf_scores, st_scores, atol=1e-2):
        hf_tok = hf_scorer.batch_tokenize(records)
        st_tok = st_scorer.batch_tokenize(records)

        print("\n--- Diagnostic Tokenization Comparison ---")
        print(f"HF input_ids shape: {hf_tok.ids.shape}")
        print(f"ST input_ids shape: {st_tok.ids.shape}")
        print(f"HF input_ids[0]: {hf_tok.ids[0]}")
        print(f"ST input_ids[0]: {st_tok.ids[0]}")

        with torch.no_grad():
            hf_with_st_tok_logits = hf_scorer(records, tokenized=st_tok).view(-1)
            if activation_fn is not None and not isinstance(
                activation_fn, nn.Identity
            ):
                hf_with_st_tok_scores = activation_fn(hf_with_st_tok_logits)
            else:
                hf_with_st_tok_scores = hf_with_st_tok_logits
        print(f"HFCrossScorer using ST tokens (scores): {hf_with_st_tok_scores}")

    # Assert match within tolerance (tokenization differences like double </s> vs single </s> may cause minor logit variations)
    assert torch.allclose(
        hf_scores.to(st_scores.dtype), st_scores, atol=1e-2
    ), f"Mismatch between HFCrossScorer ({hf_scores}) and STCrossEncoder ({st_scores})"
