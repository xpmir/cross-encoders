"""This file contains utilities for building Retrievers that can be used in various experiments"""

from typing import Any, Dict, List, Optional
from experimaestro import Launcher, stop_tags
from datamaestro_ir.data import Documents
from xpmir.rankers import Retriever
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever, SeismicSparseRetriever
from xpmir.rankers.standard import BM25, Model
from xpmir.index.bow import BOWRetriever, BOWSparseRetrieverIndexBuilder
from xpmir.index.plaid import PlaidRetriever
from xpm_torch.configuration import FabricConfiguration

from configuration import CE_FineTuning, ScorerType
from index_utils import get_splade_index, get_plaid_index
import logging

logger = logging.getLogger(__name__)

### Factories for bm25, Splade, and PLAID retrievers


def plaid_retriever(
    cfg,
    name: str,
    encoder,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    seed: int = 42,
    tags: Optional[Dict[str, Any]] = None,
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
    **kwargs,
) -> Retriever.C:
    """Factory for PLAID Retriever, given the current configuration"""

    retriever = (
        PlaidRetriever.C(
            encoder=encoder,
            index=stop_tags(
                get_plaid_index(
                    documents=documents,
                    encoder=encoder,
                    plaid_cfg=cfg.plaid,
                    launcher_index=launcher_index,
                    indexation_cfg=getattr(cfg, "indexation", None),
                    init_tasks=init_tasks,
                    seed=seed,
                    long_launcher_index=long_launcher_index,
                    long_evals=long_evals,
                    key=key,
                )
            ),
            topk=topk or cfg.retrieval.k,
            n_ivf_probe=cfg.plaid.n_ivf_probe,
            n_full_scores=cfg.plaid.n_full_scores,
            fabric_config=FabricConfiguration.C(precision="bf16-true"),
        )
        .tag("first_stage", "plaid")
        .tag("scorer", name)
        .tag("scorer_type", ScorerType.stMultiVectorEncoder.value)
        .tag("plaid_retriever", True)
    )
    if tags:
        for k, v in tags.items():
            if k not in ("first_stage", "scorer", "scorer_type", "plaid_retriever"):
                retriever.tag(k, v)
    return retriever


def splade_retriever(
    cfg: CE_FineTuning,
    name: str,
    encoder: Model,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    in_memory: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
    **kwargs,
) -> Retriever.C:
    seismic_cfg = getattr(cfg, "seismic", None)
    is_seismic = seismic_cfg is not None and getattr(seismic_cfg, "enabled", False)

    index_obj = stop_tags(  # drop 'index_documents' tag that clutters eval.
        get_splade_index(
            documents,
            splade_encoder=encoder,
            indexation_cfg=cfg.indexation,
            launcher_index=launcher_index,
            init_tasks=init_tasks,
            long_launcher_index=long_launcher_index,
            long_evals=long_evals,
            key=key,
            seismic_cfg=seismic_cfg,
        )
    )

    batch_size = cfg.retrieval.batch_size
    in_memory = getattr(cfg.retrieval, "in_memory", False) or in_memory

    if is_seismic:
        retriever = (
            SeismicSparseRetriever.C(
                index=index_obj,
                topk=topk or cfg.retrieval.k,
                batchsize=batch_size,
                encoder=encoder,
                in_memory=in_memory,
                query_cut=seismic_cfg.query_cut,
                heap_factor=seismic_cfg.heap_factor,
                n_knn=seismic_cfg.n_knn,
            )
            .tag("first_stage", name)
            .tag("approximate", "seismic")
        )
    else:
        retriever = SparseRetriever.C(
            index=index_obj,
            topk=topk or cfg.retrieval.k,
            batchsize=batch_size,
            encoder=encoder,
            in_memory=in_memory,
        ).tag("first_stage", name)
    if tags:
        for k, v in tags.items():
            if k != "first_stage":
                retriever.tag(k, v)
    return retriever


def bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index=None,
    topk: int = None,
    tags: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""

    # -----The baseline------

    bow_index = BOWSparseRetrieverIndexBuilder.C(
        documents=documents,
        max_docs=cfg.indexation.max_indexed,
    ).submit(launcher=launcher_index)

    retriever = BOWRetriever.C(
        index=stop_tags(bow_index),
        model=BM25.C(),
        topk=topk or cfg.retrieval.k,
    ).tag("first_stage", name)
    if tags:
        for k, v in tags.items():
            if k != "first_stage":
                retriever.tag(k, v)
    return retriever


def anserini_bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index,
    topk: int = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""
    index = anserini.index_builder(launcher=launcher_index)(documents)
    return anserini.AnseriniRetriever.C(
        k=topk or cfg.retrieval.k,
        model=BM25.C(),
        index=stop_tags(index),
        store=documents,
    ).tag("first_stage", name)
