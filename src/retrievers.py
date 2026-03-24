"""This file contains utilities for building Retrievers that can be used in various experiments"""

from typing import List, Dict, Optional
from datamaestro_ir.data import (
    AdhocRun,
    IDTextRecord,
    ScoredDocument,
    DocumentStore,
    Documents,
)
from xpmir.rankers import Retriever
from xpmir.evaluation import RetrieverFactory
from experimaestro import Param
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever
from xpmir.rankers.standard import BM25, Model

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


# TODO - use newer xmpir bm25 version - no need for java anymore
def bm25_retriever(
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


class RunRetriever(Retriever):
    """A retriever that returns documents from a pre-computed run"""

    run: Param[AdhocRun]
    """The pre-computed run"""

    documents: Param[Documents]
    """Associated documents"""

    def initialize(self):
        super().initialize()
        self._run_dict = self.run.get_dict()

    def collection(self):
        return self.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        qid = record["id"]
        results = self._run_dict.get(qid, {})

        # Sort by score descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Hydrate documents if the documents object is a store
        if isinstance(self.documents, DocumentStore):
            doc_ids = [doc_id for doc_id, _ in sorted_results]
            hydrated_docs = self.documents.documents_ext(doc_ids)
            return [
                ScoredDocument(doc, float(score))
                for doc, (_, score) in zip(hydrated_docs, sorted_results)
            ]

        # Fallback to only ID
        return [
            ScoredDocument({"id": doc_id}, float(score))
            for doc_id, score in sorted_results
        ]

    def _store(self) -> Optional[DocumentStore]:
        return self.documents if isinstance(self.documents, DocumentStore) else None


class MultiRunRetrieverFactory(RetrieverFactory):
    """A factory that returns the appropriate RunRetriever for a given dataset"""

    def __init__(self, retriever_name: str):
        self.retriever_name = retriever_name
        self.runs: Dict[str, AdhocRun] = {}
        self.documents: Dict[str, Documents] = {}

    def add_run(self, key: str, documents: Documents, run: AdhocRun):
        """Register a run for a given document collection"""
        if key in self.runs.keys():
            logger.warning(
                f"{key} Retrival run already stored for {self.retriever_name}"
            )
        self.runs[key] = run
        self.documents[key] = documents

    def __call__(self, dataset: Documents, key: str = None) -> RunRetriever:
        # Try to find the run by key first, then by dataset ID
        run = self.runs.get(key) if key else None
        if run is None:
            # Fallback to dataset ID if key not provided or not found
            # This is less specific but better than nothing
            for k, docs in self.documents.items():
                if docs.id == dataset.id:
                    run = self.runs[k]
                    break

        if run is None:
            raise KeyError(
                f"No run found for dataset key='{key}' or id='{dataset.id}'"
                f"Available: {','.join(self.runs.keys())}"
            )

        return RunRetriever.C(run=run, documents=dataset).tag(
            "first_stage", self.retriever_name
        )

    @classmethod
    def from_results(cls, name: str, results: List) -> "MultiRunRetrieverFactory":
        factory = cls(name)
        for res in results:
            factory.add_run(res.key, res.task.dataset.documents, res.run)
        return factory
