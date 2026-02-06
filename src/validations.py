
from functools import lru_cache

from xpmir.datasets.adapters import RandomFold

from xpmir.papers.helpers.samplers import ValidationSample
from xpmir.papers.helpers.samplers import prepare_collection

import logging
logger = logging.getLogger(__name__)

NANO_BEIR = [
    'arguana', 'climate-fever', 'dbpedia-entity', 'fever', 
    'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 
    'quora', 'scidocs', 'scifact', 'webis-touche2020'
]

@lru_cache
def nanobeir_validation_datasets(
    cfg: ValidationSample, launcher=None
):
    """Sample dev topics to get a validation subset on the NFCorpus dataset."""

    random_folds = {}
    documents = {}

    for dataset_name in NANO_BEIR:
        # Prepare dataset components
        dataset = prepare_collection(f'irds.nano-beir.{dataset_name}')
        print(f"Loaded: {dataset_name}")

        random_folds[dataset_name] = RandomFold.C(
            dataset=dataset,
            seed=cfg.seed,
            fold=0,
            sizes=[cfg.size],
        ).submit(launcher=launcher)
        documents[dataset_name] = dataset.documents
        _ = next(dataset.instance().documents.iter_documents()) # Force load documents
        _ = next(dataset.instance().topics.iter()) # Force load queries

    return random_folds, documents