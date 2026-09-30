"""Utilities for index management."""

import logging
from typing import Optional
from xpmir.index.sparse import SparseRetrieverIndexBuilder
from xpmir.index.plaid import PlaidIndexBuilder
from xpmir.rankers import Documents
from xpm_torch.configuration import FabricConfiguration
from configuration import Indexation, PlaidConfiguration

logger = logging.getLogger(__name__)

# cache the indexes
_indexes = {}


def get_splade_index(
    documents: Documents,
    splade_encoder,
    indexation_cfg,
    launcher_index,
    init_tasks: list = [],
):
    """Build an index for given documents, using a given Sparse retriever model
    Caches it to avoid submitting job twice.
    """

    index_cfg = SparseRetrieverIndexBuilder.C(
        batch_size=indexation_cfg.batch_size,
        # batcher=PowerAdaptativeBatcher.C(),
        encoder=splade_encoder,
        documents=documents,
        ordered_index=False,
        max_docs=indexation_cfg.max_indexed,
    )

    indexer_id = index_cfg.__identifier__()

    if indexer_id not in _indexes:
        logger.info(
            "Indexing %s (%s documents) with %s",
            documents.id,
            documents.count,
            launcher_index,
        )
        index = index_cfg.tag("index_documents", documents.id).submit(
            launcher=launcher_index, init_tasks=init_tasks
        )
        _indexes[indexer_id] = index
    else:
        index = _indexes[indexer_id]

    return index


def get_plaid_index(
    documents: Documents,
    encoder,
    plaid_cfg: PlaidConfiguration,
    launcher_index,
    indexation_cfg: Optional[Indexation] = None,
    batch_size: Optional[int] = None,
    init_tasks: list = None,
    seed: int = 42,
):
    """Build a PLAID index for given documents using a given multi-vector/late-interaction encoder.
    Caches it to avoid submitting the job twice.
    """
    if init_tasks is None:
        init_tasks = []

    # Document encoder batch size (GPU forward pass)
    if batch_size is not None and batch_size > 0:
        doc_batch_size = batch_size
    elif indexation_cfg is not None and indexation_cfg.batch_size > 0:
        doc_batch_size = indexation_cfg.batch_size
    else:
        doc_batch_size = 32

    # Fast-PLAID internal token batch size
    fast_plaid_bs = (
        plaid_cfg.fast_plaid_batch_size
        if plaid_cfg.fast_plaid_batch_size > 0
        else 25000
    )

    index_builder = PlaidIndexBuilder.C(
        documents=documents,
        encoder=encoder,
        buffer_size=plaid_cfg.buffer_size,
        batch_size=doc_batch_size,
        fast_plaid_batch_size=fast_plaid_bs,
        n_bits=plaid_cfg.n_bits,
        kmeans_niters=plaid_cfg.kmeans_niters,
        n_samples_kmeans=plaid_cfg.n_samples_kmeans,
        max_points_per_centroid=plaid_cfg.max_points_per_centroid,
        seed=seed,
        compress_only=plaid_cfg.compress_only,
        force_cpu_indexing=plaid_cfg.force_cpu_indexing,
        fabric_config=FabricConfiguration.C(precision="bf16-true"),
    )

    init_paths = tuple(str(getattr(t, "path", t)) for t in init_tasks)
    cache_key = (index_builder.__identifier__(), init_paths)

    if cache_key not in _indexes:
        logger.info(
            "Indexing %s (%s documents) with PLAID via %s",
            documents.id,
            getattr(documents, "count", "unknown"),
            launcher_index,
        )
        index = index_builder.tag("index_documents", documents.id).submit(
            launcher=launcher_index, init_tasks=init_tasks
        )
        _indexes[cache_key] = index
    else:
        index = _indexes[cache_key]

    return index
