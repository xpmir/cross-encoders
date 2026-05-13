from functools import lru_cache
from datamaestro import prepare_dataset

from xpmir.datasets.adapters import RandomFold
from xpmir.datasets.samplers import prepare_collection
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.measures import RR, nDCG, R, Success
from configuration import Evaluation

import logging

logger = logging.getLogger(__name__)


def check_datasets_docs(evaluations_collection: EvaluationsCollection):
    """Ensure that documents exists -> triggers any lazy loading issues now
    ir_datasets show download them with prepare_dataset ... but not load them until accessed
    """
    for evals in evaluations_collection.collection.values():
        try:
            _ = next(evals.dataset.documents.iter_documents())
        except FileNotFoundError as e:
            logger.error(
                f"{e}- cannot dowload {evals.dataset.documents.id}, please consider adding it manually (may happen for proprietary datasets such as Robust04)"
            )


CE_MEASURES = [Success @ 5, nDCG @ 10, nDCG @ 20, RR @ 10]
RETRIEVERS_MEASURES = [R @ 100, R @ 1000]


def get_fold(dataset, size, seed=0, launcher=None):
    if size > 0:
        (fold_config,) = RandomFold.folds(
            seed=seed, sizes=[size], dataset=dataset, submit=False
        )
        return fold_config.submit(launcher=launcher)
    return dataset


@lru_cache
def minified_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to use for testing.
    As of now, this list includes:
    - MS Marco v1 devsmall (with a reduced number of topics)
    - TREC DL 2019
    - TREC DL 2020
    - SciFact
    - Touché-2020
    - FiQA-2018
    - NFCorpus
    """

    # dl21 = prepare_dataset("irds.msmarco-passage-v2.trec-dl-2021.judged")
    # dl21.documents.file_access = FileAccess.FILE
    # dl22 = prepare_dataset("irds.msmarco-passage-v2.trec-dl-2022.judged")
    # dl22.documents.file_access = FileAccess.FILE
    # return EvaluationsCollection(
    #     trec2021=Evaluations(dl21, CE_MEASURES),
    #     trec2022=Evaluations(dl22, CE_MEASURES),
    # )
    v1_devsmall_ds = prepare_collection("com.microsoft.msmarco.passage.dev.small")
    dl19 = prepare_dataset("com.microsoft.msmarco.passage.trec2019.judged")
    dl20 = prepare_dataset("com.microsoft.msmarco.passage.trec2020.judged")

    v1_devsmall_ds = get_fold(v1_devsmall_ds, test_topic_nb, launcher=launcher)

    scifact = prepare_dataset("org.beir.scifact.test")  # 300 queries
    scifact = get_fold(scifact, test_topic_nb, launcher=launcher)

    touche = prepare_dataset(
        "org.beir.webis.touche2020.v2"
    )  # v2 as it fixes some of v1 issues

    fiqa = prepare_dataset("org.beir.fiqa.test")  # 648 queries
    fiqa = get_fold(fiqa, test_topic_nb, launcher=launcher)

    nfcorpus = prepare_dataset("org.beir.nfcorpus.test")  # 323 queries
    nfcorpus = get_fold(nfcorpus, test_topic_nb, launcher=launcher)

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    tests = EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, measures=measures),
        trec2019=Evaluations(dl19, measures=measures),
        trec2020=Evaluations(dl20, measures=measures),
        scifact=Evaluations(scifact, measures=measures),
        touche=Evaluations(touche, measures=measures),
        fiqa=Evaluations(fiqa, measures=measures),
        nfcorpus=Evaluations(nfcorpus, measures=measures),
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(tests)

    return tests


@lru_cache
def BEIR_13_tests(
    test_topic_nb: int, retrievers_only: bool = False, launcher=None
) -> EvaluationsCollection:
    """All of BEIR (minus the 5 datasets not publicly available)
    - ArguAna
    - Climate-FEVER
    - DBPedia
    - FEVER
    - FiQA-2018
    - HotPotQA
    - NFCorpus
    - NQ
    - Quora
    - SciDocs
    - SciFact
    - TREC-COVID
    - Touché-2020
    """

    ## BEIR datasets
    scifact = prepare_dataset("org.beir.scifact.test")  # 300 queries
    scifact = get_fold(scifact, test_topic_nb, launcher=launcher)

    touche = prepare_dataset("org.beir.webis.touche2020.v2")
    # v2 as it fixes some of v1 issues

    fiqa = prepare_dataset("org.beir.fiqa.test")  # 648 queries
    fiqa = get_fold(fiqa, test_topic_nb, launcher=launcher)

    nfcorpus = prepare_dataset("org.beir.nfcorpus.test")  # 323 queries
    nfcorpus = get_fold(nfcorpus, test_topic_nb, launcher=launcher)

    arguana = prepare_dataset("org.beir.arguana")
    arguana = get_fold(arguana, test_topic_nb, launcher=launcher)

    climate_fever = prepare_dataset("org.beir.climate.fever")
    climate_fever = get_fold(climate_fever, test_topic_nb, launcher=launcher)

    dbpedia = prepare_dataset("org.beir.dbpedia.entity.test")
    dbpedia = get_fold(dbpedia, test_topic_nb, launcher=launcher)

    fever = prepare_dataset("org.beir.fever.test")
    fever = get_fold(fever, test_topic_nb, launcher=launcher)

    hotpotqa = prepare_dataset("org.beir.hotpotqa.test")
    hotpotqa = get_fold(hotpotqa, test_topic_nb, launcher=launcher)

    nq = prepare_dataset("org.beir.nq")
    nq = get_fold(nq, test_topic_nb, launcher=launcher)

    quora = prepare_dataset("org.beir.quora.test")
    quora = get_fold(quora, test_topic_nb, launcher=launcher)

    scidocs = prepare_dataset("org.beir.scidocs")
    scidocs = get_fold(scidocs, test_topic_nb, launcher=launcher)

    trec_covid = prepare_dataset("org.beir.trec.covid")
    trec_covid = get_fold(trec_covid, test_topic_nb, launcher=launcher)

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    return EvaluationsCollection(
        fever=Evaluations(fever, measures=measures),
        arguana=Evaluations(arguana, measures=measures),
        climate_fever=Evaluations(climate_fever, measures=measures),
        dbpedia=Evaluations(dbpedia, measures=measures),
        fiqa=Evaluations(fiqa, measures=measures),
        hotpotqa=Evaluations(hotpotqa, measures=measures),
        nfcorpus=Evaluations(nfcorpus, measures=measures),
        nq=Evaluations(nq, measures=measures),
        quora=Evaluations(quora, measures=measures),
        scidocs=Evaluations(scidocs, measures=measures),
        scifact=Evaluations(scifact, measures=measures),
        touche=Evaluations(touche, measures=measures),
        trec_covid=Evaluations(trec_covid, measures=measures),
    )


@lru_cache
def Robust04_test(
    test_topic_nb: int, retrievers_only: bool = False, launcher=None
) -> EvaluationsCollection:
    """Robust04 dataset"""
    logger.info("Preparing Robust04 dataset...")
    robust04 = prepare_dataset("gov.nist.trec.adhoc.robust.2004.withstore")
    robust04 = get_fold(robust04, test_topic_nb, launcher=launcher)

    return EvaluationsCollection(
        robust04=Evaluations(
            robust04, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
    )


@lru_cache
def LoTTE_tests(
    test_topic_nb: int, retrievers_only: bool = False, launcher=None
) -> EvaluationsCollection:
    """LoTTE Search dataset"""
    logger.info("Preparing LoTTE Search datasets...")
    lotte_writing = prepare_dataset("edu.stanford.lotte.writing.test.search")
    lotte_writing = get_fold(lotte_writing, test_topic_nb, launcher=launcher)

    lotte_recreation = prepare_dataset("edu.stanford.lotte.recreation.test.search")
    lotte_recreation = get_fold(lotte_recreation, test_topic_nb, launcher=launcher)

    lotte_science = prepare_dataset("edu.stanford.lotte.science.test.search")
    lotte_science = get_fold(lotte_science, test_topic_nb, launcher=launcher)

    lotte_technology = prepare_dataset("edu.stanford.lotte.technology.test.search")
    lotte_technology = get_fold(lotte_technology, test_topic_nb, launcher=launcher)

    lotte_lifestyle = prepare_dataset("edu.stanford.lotte.lifestyle.test.search")
    lotte_lifestyle = get_fold(lotte_lifestyle, test_topic_nb, launcher=launcher)

    return EvaluationsCollection(
        lotte_writing=Evaluations(
            lotte_writing, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        lotte_recreation=Evaluations(
            lotte_recreation,
            CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES,
        ),
        lotte_science=Evaluations(
            lotte_science, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        lotte_technology=Evaluations(
            lotte_technology,
            CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES,
        ),
        lotte_lifestyle=Evaluations(
            lotte_lifestyle, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
    )


NANO_BEIR_IDS = [
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "webis-touche2020",
]

# Nano Beir Datamaestro Ids from id
NANO_BEIR_KEYS = {
    f"nano_{name}": f"co.huggingface.nano-beir.{name}" for name in NANO_BEIR_IDS
}


@lru_cache
def nano_beir_tests(
    test_topic_nb: int, retrievers_only: bool = False, launcher=None
) -> EvaluationsCollection:
    """NanoBEIR datasets"""

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    evals = {}
    for key, dataset_id in NANO_BEIR_KEYS.items():
        # Map back to the dataset name (e.g., nano_arguana -> arguana)
        ds = prepare_dataset(dataset_id)
        ds = get_fold(ds, test_topic_nb, launcher=launcher)
        evals[key] = Evaluations(ds, measures=measures)

    return EvaluationsCollection(**evals)


@lru_cache
def paper_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to include in the paper.
    As of now, this list includes all of BEIR (minus the 5 datasets not publicly available)
    + LoTTE (Search)
    + the 2 TREC-DL 19 and 20 datasets, i.e.:
    - MS Marco v1 (dev set)
    - TREC DL 2019
    - TREC DL 2020
    - BEIR 13:
        - ArguAna
        - Climate-FEVER
        - DBPedia
        - FEVER
        - FiQA-2018
        - HotPotQA
        - NFCorpus
        - NQ
        - Quora
        - SciDocs
        - SciFact
        - TREC-COVID
        - Touché-2020
    """

    # In domain - MS Marco + TREC DL
    v1_dev = prepare_collection("com.microsoft.msmarco.passage.dev.small")
    dl19 = prepare_dataset("com.microsoft.msmarco.passage.trec2019.judged")
    dl20 = prepare_dataset("com.microsoft.msmarco.passage.trec2020.judged")

    v1_dev = get_fold(v1_dev, test_topic_nb, launcher=launcher)
    dl19 = get_fold(dl19, test_topic_nb, launcher=launcher)
    dl20 = get_fold(dl20, test_topic_nb, launcher=launcher)

    # Out of domain - BEIR
    beir = BEIR_13_tests(
        test_topic_nb, retrievers_only=retrievers_only, launcher=launcher
    )
    robust04 = Robust04_test(
        test_topic_nb, retrievers_only=retrievers_only, launcher=launcher
    )
    lotte = LoTTE_tests(
        test_topic_nb, retrievers_only=retrievers_only, launcher=launcher
    )

    paper_tests_res = EvaluationsCollection(
        msmarco_dev=Evaluations(
            v1_dev, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        trec2019=Evaluations(
            dl19, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        trec2020=Evaluations(
            dl20, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        **beir.collection,
        **robust04.collection,
        **lotte.collection,
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(paper_tests_res)
    return paper_tests_res


def build_tests(
    cfg: Evaluation,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
) -> EvaluationsCollection:
    """Build the tests to use for evaluation during training or at the end of it.
    :param cfg: Configuration for the evaluation
    :param check_docs: Whether to check that documents are accessible (triggers downloads if needed)
    :returns: The evaluations collection to use
    """
    all_evals = {}

    # Helper to add evals to the dictionary
    def add_evals(evals_collection: EvaluationsCollection):
        for name, evals in evals_collection.collection.items():
            if name not in all_evals:
                all_evals[name] = evals

    # 2. In-domain (MSMarco + TREC DL)
    if cfg.in_domain or cfg.all_datasets:
        v1_dev = prepare_collection("com.microsoft.msmarco.passage.dev.small")
        dl19 = prepare_dataset("com.microsoft.msmarco.passage.trec2019.judged")
        dl20 = prepare_dataset("com.microsoft.msmarco.passage.trec2020.judged")

        v1_dev = get_fold(v1_dev, cfg.test_max_topics, launcher=launcher)
        dl19 = get_fold(dl19, cfg.test_max_topics, launcher=launcher)
        dl20 = get_fold(dl20, cfg.test_max_topics, launcher=launcher)

        measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        add_evals(
            EvaluationsCollection(
                msmarco_dev=Evaluations(v1_dev, measures),
                trec2019=Evaluations(dl19, measures),
                trec2020=Evaluations(dl20, measures),
            )
        )

    # 3. BEIR13 (All of it)
    if cfg.beir13 or cfg.all_datasets:
        add_evals(
            BEIR_13_tests(
                cfg.test_max_topics, retrievers_only=retrievers_only, launcher=launcher
            )
        )

    if cfg.robust04 or cfg.all_datasets:
        add_evals(
            Robust04_test(
                cfg.test_max_topics, retrievers_only=retrievers_only, launcher=launcher
            )
        )

    if cfg.lotte_search or cfg.all_datasets:
        add_evals(
            LoTTE_tests(
                cfg.test_max_topics, retrievers_only=retrievers_only, launcher=launcher
            )
        )

    # 1. NanoBEIR
    if cfg.nanobeir:
        add_evals(
            nano_beir_tests(
                cfg.test_max_topics, retrievers_only=retrievers_only, launcher=launcher
            )
        )
    # 4. Specific datasets
    if cfg.datasets:
        # We need a way to map dataset names to their respective prepare functions
        # For now, let's look them up in BEIR_13_tests if possible, or handle individually
        beir13_all = BEIR_13_tests(
            cfg.test_max_topics, retrievers_only=retrievers_only, launcher=launcher
        )
        for ds_name in cfg.datasets:
            if ds_name in beir13_all.collection:
                if ds_name not in all_evals:
                    all_evals[ds_name] = beir13_all.collection[ds_name]
            else:
                logger.warning(
                    f"Dataset {ds_name} not found in BEIR13 or supported list"
                )

    # If nothing was selected, use minified_tests as default (original behavior)
    if not all_evals and not cfg.all_datasets:
        return minified_tests(
            cfg.test_max_topics,
            check_docs=check_docs,
            retrievers_only=retrievers_only,
            launcher=launcher,
        )

    tests = EvaluationsCollection(**all_evals)

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(tests)

    return tests


def get_max_query_length(
    tokenizer, evaluations_collection: EvaluationsCollection
) -> dict[str, int]:
    """Helper to find the maximum query length for each dataset in the collection."""
    from xpmir.text.huggingface.tokenizers import HFTokenizer

    res = {}

    # Get the underlying transformers tokenizer
    if isinstance(tokenizer, HFTokenizer):
        if not hasattr(tokenizer, "tokenizer"):
            tokenizer.__initialize__()
        hf_tokenizer = tokenizer.tokenizer
    else:
        hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    for name, evals in evaluations_collection.collection.items():
        unique_queries = {
            topic["text_item"].text for topic in evals.dataset.topics.iter_topics()
        }

        if not unique_queries:
            res[name] = 0
            continue

        # Tokenize all queries and find the maximum length
        lengths = [
            len(hf_tokenizer.encode(q, add_special_tokens=False))
            for q in unique_queries
        ]
        res[name] = max(lengths)

    return res
