from functools import lru_cache

from xpmir.datasets.adapters import RandomFold

from xpmir.papers.helpers.samplers import ValidationSample
from xpmir.papers.helpers.samplers import prepare_collection

from tests import NANO_BEIR_KEYS

import logging

logger = logging.getLogger(__name__)


@lru_cache
def nano_msmarco_validation_datasets(cfg: ValidationSample, launcher=None):
    """Return validation over msmarco."""

    dataset = prepare_collection("co.huggingface.nano-beir.msmarco")
    logger.info("Loaded: msmarco")
    _ = next(dataset.documents.iter_documents())  # Force load documents
    _ = next(dataset.topics.iter())  # Force load queries

    random_folds = RandomFold.C(
        dataset=dataset,
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
    ).submit(launcher=launcher)

    return random_folds, dataset.documents


@lru_cache
def nanobeir_validation_datasets(cfg: ValidationSample, launcher=None):
    """Return validations over all the NANO_BEIR datasets."""

    random_folds = {}
    documents = {}

    for dataset_name, dataset_id in NANO_BEIR_KEYS.items():
        # Prepare dataset components
        dataset = prepare_collection(dataset_id)
        logger.info(f"Loaded: {dataset_name}")

        random_folds[dataset_name] = RandomFold.C(
            dataset=dataset,
            seed=cfg.seed,
            fold=0,
            sizes=[cfg.size],
        ).submit(launcher=launcher)
        documents[dataset_name] = dataset.documents

        # Force load documents, queries, and qrels (if available)
        _ = next(dataset.documents.iter_documents())
        _ = next(dataset.topics.iter())
        if hasattr(dataset, "assessments"):
            _ = dataset.assessments.iter()
    return random_folds, documents
