import torch
import pytest
from sentence_transformers import CrossEncoder
from xpmir.neural.sentence_transformers import st_cross_scorer
from xpmir.letor.records import PointwiseItems
import logging

@pytest.mark.parametrize(
    "model_id",
    [
        # "Qwen/Qwen3-Reranker-0.6B",
        # "mixedbread-ai/mxbai-rerank-base-v2",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    ],
)
def test_st_cross_encoder_parity(model_id):
    """Verify parity between raw sentence-transformers and STCrossEncoder.
    Ensures that both direct and tokenized calls in xpmir yield the same scores as the raw ST implementation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # 1. Setup raw ST implementation
    model = CrossEncoder(model_id, device=str(device))

    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]

    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    ]

    pairs = [[query, doc] for query, doc in zip(queries, documents)]

    # Process one by one to avoid ST's internal batch sorting for exact parity
    raw_scores = []
    for pair in pairs:
        score = model.predict([pair], convert_to_tensor=True)
        raw_scores.append(score)
    raw_scores = torch.cat(raw_scores)

    # 2. Setup xpmir STCrossEncoder implementation
    scorer = st_cross_scorer(model_id=model_id)
    scorer = scorer.instance()
    scorer.to(device)
    scorer.initialize()

    # 3. Compare scores
    for i in range(len(queries)):
        records = PointwiseItems.from_texts(
            topics=[queries[i]], documents=[documents[i]]
        )

        # Direct call
        with torch.no_grad():
            score_dir = scorer(records).view(-1)

        # Pre-tokenized call
        tokenized = scorer.batch_tokenize(records)
        with torch.no_grad():
            score_tok = scorer(records, tokenized=tokenized).view(-1)

        # Assertions
        expected = raw_scores[i]
        # Use a slightly higher tolerance for BFloat16 vs Float32 if necessary
        # but Qwen3-Reranker-0.6B usually yields exact matches in sequential mode
        assert torch.allclose(
            score_dir.to(expected.dtype), expected, atol=1e-4
        ), f"Direct score mismatch at index {i}: expected {expected}, got {score_dir}"
        assert torch.allclose(
            score_tok.to(expected.dtype), expected, atol=1e-4
        ), f"Tokenized score mismatch at index {i}: expected {expected}, got {score_tok}"


def test_st_cross_encoder_hf_query_doc_tokenizer():
    """Verify STCrossEncoder using HFQueryDocTokenizer with custom query/doc length limits."""
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    scorer = st_cross_scorer(
        model_id=model_id,
        max_length=128,
        max_query_length=16,
        max_doc_length=64,
    )
    scorer = scorer.instance()
    scorer.initialize()

    records = PointwiseItems.from_texts(
        topics=["What is the speed of light in vacuum?" * 5],
        documents=[
            "The speed of light in vacuum is exactly 299,792,458 meters per second." * 10
        ],
    )

    tokenized = scorer.batch_tokenize(records)
    assert tokenized.ids.shape[1] <= 128
    scores = scorer(records, tokenized=tokenized)
    assert scores.shape == (1,)


def test_st_cross_encoder_main_process_initialization():
    """Verify that STCrossEncoder.initialize() instantiates st_model in main process."""
    scorer = st_cross_scorer(model_id="cross-encoder/ms-marco-MiniLM-L6-v2")
    scorer = scorer.instance()

    assert scorer.st_model is None
    scorer.initialize()
    assert scorer._initialized is True
    assert scorer.st_model is not None


@pytest.mark.parametrize(
    "model_id",
    [
        # "Qwen/Qwen3-Reranker-0.6B",
        # "mixedbread-ai/mxbai-rerank-base-v2",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    ],
)
def test_init_distilled_st_cross_encoder(model_id: str):
    """Verify InitDistilledSTCrossEncoder sets _initialized = True and respects custom st_model."""
    from cross_encoders_training.experiment import InitDistilledSTCrossEncoder
    class DummyConfig:
        base = model_id
        max_length = 64
        pref_attn_implementation = "sdpa"

    cfg = DummyConfig()
    scorer = st_cross_scorer(model_id=cfg.base, max_length=cfg.max_length)
    init_task = InitDistilledSTCrossEncoder.C(model=scorer, pref_attn_implementation="sdpa")

    task_inst = init_task.instance()
    task_inst.execute()
    scorer_inst = task_inst.model

    logging.debug(f"Created model: {scorer_inst}")
    assert scorer_inst._initialized is True
    assert scorer_inst.st_model is not None

    # Secondary call to initialize() must not overwrite custom st_model
    custom_model = scorer_inst.st_model
    scorer_inst.initialize()
    assert scorer_inst.st_model is custom_model


def test_uninitialized_st_cross_encoder_raises_error():
    """Verify that accessing methods on an uninitialized STCrossEncoder raises RuntimeError."""
    scorer = st_cross_scorer(model_id="cross-encoder/ms-marco-MiniLM-L6-v2")
    scorer = scorer.instance()

    records = PointwiseItems.from_texts(topics=["query"], documents=["doc"])
    with pytest.raises(RuntimeError, match="has not been initialized"):
        scorer(records)

    with pytest.raises(RuntimeError, match="has not been initialized"):
        scorer.tokenize(records)


def test_hf_query_doc_tokenizer_flattening():
    """Verify HFQueryDocTokenizer flattening pre-instantiates collator and produces FA2 kwargs."""
    from xpmir.neural.huggingface import HFQueryDocTokenizer

    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    tok = HFQueryDocTokenizer.C(
        model_id=model_id,
        max_length=128,
        max_query_length=16,
        max_doc_length=64,
        flatten=True,
    ).instance()

    assert tok.data_collator is None
    tok.initialize()
    assert tok.data_collator is not None
    collator_ref = tok.data_collator

    records = PointwiseItems.from_texts(
        topics=["What is the speed of light in vacuum?", "Short query"],
        documents=[
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "Short document text.",
        ],
    )

    tokenized = tok.tokenize(records)
    # Reused collator without re-instantiation
    assert tok.data_collator is collator_ref

    # Flattened format assertions
    assert tokenized.ids.ndim == 2
    assert tokenized.ids.shape[0] == 1  # 1D flattened batch of shape (1, total_tokens)
    assert tokenized.mask is None
    assert tokenized.token_type_ids is None
    assert tokenized.kwargs is not None

    required_keys = {"cu_seq_lens_q", "cu_seq_lens_k", "position_ids", "seq_idx", "max_length_q"}
    assert required_keys.issubset(tokenized.kwargs.keys())
    assert tokenized.kwargs["cu_seq_lens_q"].shape == (3,)  # batch_size + 1
    assert tokenized.kwargs["position_ids"].shape == tokenized.ids.shape
    assert tokenized.kwargs["seq_idx"].shape == tokenized.ids.shape


def test_st_cross_encoder_with_flattened_qd_tokenizer():
    """Verify STCrossEncoder runs seamlessly with flattened HFQueryDocTokenizer without modifying forward."""
    from cross_encoders_training.experiment import build_STCrossEncoder

    class DummyConfig:
        base = "cross-encoder/ms-marco-MiniLM-L6-v2"
        max_length = 128
        max_query_length = 16
        max_doc_length = 64
        pref_attn_implementation = "sdpa"

    cfg = DummyConfig()
    scorer, init_tasks = build_STCrossEncoder(cfg)
    scorer = scorer.instance()

    # Force flattening on the tokenizer to simulate FA2 varlen collation
    scorer.tokenizer.flatten = True

    # Execute init task (InitDistilledSTCrossEncoder)
    init_task = init_tasks[0].instance()
    init_task.model = scorer
    init_task.execute()

    assert scorer.tokenizer.data_collator is not None

    records = PointwiseItems.from_texts(
        topics=["What is the speed of light in vacuum?", "Who was Isaac Newton?"],
        documents=[
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "Sir Isaac Newton was an English mathematician and physicist.",
        ],
    )

    # Tokenize through the model
    tokenized = scorer.batch_tokenize(records)
    assert tokenized.ids.shape[0] == 1
    assert tokenized.mask is None
    assert "cu_seq_lens_q" in tokenized.kwargs

    # Forward through the model
    with torch.no_grad():
        scores = scorer(records, tokenized=tokenized)

    assert scores.shape == (2,)
    assert not torch.isnan(scores).any()
