"""This file contains utilities for building Retrievers that can be used in various experiments"""

from datamaestro_ir.data import Documents
from xpmir.rankers import Retriever
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever
from xpmir.rankers.standard import BM25, Model
from xpmir.index.bow import BOWRetriever, BOWSparseRetrieverIndexBuilder
from configuration import CE_FineTuning
from index_utils import get_splade_index
import logging

logger = logging.getLogger(__name__)

### Factories for bm25 and Splade retrievers


def splade_retriever(
    cfg: CE_FineTuning,
    name: str,
    encoder: Model,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    in_memory: bool = False,
    **kwargs,
) -> Retriever.C:
    """Factory for Splade Retriever, given the current configuration"""

    return SparseRetriever.C(
        index=get_splade_index(
            documents,
            splade_encoder=encoder,
            indexation_cfg=cfg.indexation,
            launcher_index=launcher_index,
            init_tasks=init_tasks,
        ),
        topk=topk or cfg.retrieval.k,
        batchsize=1,
        encoder=encoder,
        in_memory=in_memory,
    ).tag("first_stage", name)


def bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index,
    topk: int = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""

    # -----The baseline------

    bow_index = BOWSparseRetrieverIndexBuilder.C(
        documents=documents,
        max_docs=cfg.indexation.max_indexed,
    ).submit(launcher=launcher_index)

    return BOWRetriever.C(
        index=bow_index,
        model=BM25.C(),
        topk=topk or cfg.retrieval.k,
    ).tag("first_stage", name)


def anserini_bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index,
    topk: int = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""
    return anserini.AnseriniRetriever.C(
        k=topk or cfg.retrieval.k,
        model=BM25.C(),
        index=anserini.index_builder(launcher=launcher_index)(documents),
        store=documents,
    ).tag("first_stage", name)
