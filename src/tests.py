from functools import lru_cache
from datamaestro import prepare_dataset

from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.measures import AP, RR, nDCG
from xpmir.papers.helpers.samplers import prepare_collection
from configuration import Evaluation

import logging
logger = logging.getLogger(__name__)


def check_datasets_docs(evaluations_collection: EvaluationsCollection):
    """Ensure that documents exists -> triggers any lazy loading issues now
    ir_datasets show download them with prepare_dataset ... but not load them until accessed
    """
    for evals in evaluations_collection.collection.values():
        _ = next(evals.dataset.documents.iter_documents())


MEASURES = [AP, nDCG @ 10, RR @ 10]

@lru_cache
def minified_tests(test_topic_nb: int, check_docs: bool = True) -> EvaluationsCollection:
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
    #     trec2021=Evaluations(dl21, MEASURES),
    #     trec2022=Evaluations(dl22, MEASURES),
    # )
    v1_devsmall_ds = prepare_collection("irds.msmarco-passage.dev.small")
    dl19 = prepare_dataset("irds.msmarco-passage.trec-dl-2019.judged")
    dl20 = prepare_dataset("irds.msmarco-passage.trec-dl-2020.judged")
    if test_topic_nb > 0:
        (v1_devsmall_ds,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=v1_devsmall_ds
        )

    scifact = prepare_dataset("irds.beir.scifact.test") # 300 queries
    if test_topic_nb > 0:
        (scifact,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=scifact
        )
    touche = prepare_dataset("irds.beir.webis-touche2020.v2") # v2 as it fixes some of v1 issues

    fiqa = prepare_dataset("irds.beir.fiqa.test") # 648 queries
    if test_topic_nb > 0:
        (fiqa,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=fiqa
        )

    nfcorpus = prepare_dataset("irds.beir.nfcorpus.test") # 323 queries
    if test_topic_nb > 0:
        (nfcorpus,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=nfcorpus
        )

    tests =  EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, MEASURES),
        trec2019=Evaluations(dl19, MEASURES),
        trec2020=Evaluations(dl20, MEASURES),
        scifact=Evaluations(scifact, MEASURES),
        touche=Evaluations(touche, MEASURES),
        fiqa=Evaluations(fiqa, MEASURES),
        nfcorpus=Evaluations(nfcorpus, MEASURES),
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(tests)

    return tests

@lru_cache
def BEIR_tests(test_topic_nb: int) -> EvaluationsCollection:
    """ All of BEIR (minus the 5 datasets not publicly available) 
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
    scifact = prepare_dataset("irds.beir.scifact.test") # 300 queries
    if test_topic_nb > 0:
        (scifact,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=scifact
        )
    touche = prepare_dataset("irds.beir.webis-touche2020.v2") # v2 as it fixes some of v1 issues

    fiqa = prepare_dataset("irds.beir.fiqa.test") # 648 queries
    if test_topic_nb > 0:
        (fiqa,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=fiqa
        )

    nfcorpus = prepare_dataset("irds.beir.nfcorpus.test") # 323 queries
    if test_topic_nb > 0:
        (nfcorpus,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=nfcorpus
        )

    arguana = prepare_dataset("irds.beir.arguana")
    if test_topic_nb > 0:
        (arguana,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=arguana
        )

    climate_fever = prepare_dataset("irds.beir.climate-fever")
    if test_topic_nb > 0:
        (climate_fever,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=climate_fever
        )
    
    dbpedia = prepare_dataset("irds.beir.dbpedia-entity.test")
    if test_topic_nb > 0:
        (dbpedia,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=dbpedia
        )

    fever = prepare_dataset("irds.beir.fever.test")
    if test_topic_nb > 0:
        (fever,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=fever
        )

    hotpotqa = prepare_dataset("irds.beir.hotpotqa.test")
    if test_topic_nb > 0:
        (hotpotqa,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=hotpotqa
        )

    nq = prepare_dataset("irds.beir.nq")
    if test_topic_nb > 0:
        (nq,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=nq
        )

    quora = prepare_dataset("irds.beir.quora.test")
    if test_topic_nb > 0:
        (quora,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=quora
        )

    scidocs = prepare_dataset("irds.beir.scidocs")
    if test_topic_nb > 0:
        (scidocs,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=scidocs
        )

    trec_covid = prepare_dataset("irds.beir.trec-covid")
    if test_topic_nb > 0:
        (trec_covid,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=trec_covid
        )
    return EvaluationsCollection(
        fever=Evaluations(fever, MEASURES),
        arguana=Evaluations(arguana, MEASURES),
        climate_fever=Evaluations(climate_fever, MEASURES),
        dbpedia=Evaluations(dbpedia, MEASURES),
        fiqa=Evaluations(fiqa, MEASURES),
        hotpotqa=Evaluations(hotpotqa, MEASURES),
        nfcorpus=Evaluations(nfcorpus, MEASURES),
        nq=Evaluations(nq, MEASURES),
        quora=Evaluations(quora, MEASURES),
        scidocs=Evaluations(scidocs, MEASURES),
        scifact=Evaluations(scifact, MEASURES),
        touche=Evaluations(touche, MEASURES),
        trec_covid=Evaluations(trec_covid, MEASURES),
    )

@lru_cache
def Robust04_test(test_topic_nb: int) -> EvaluationsCollection:
    """ Robust04 dataset """
    robust04 = prepare_dataset("irds.disks45.nocr.trec-robust-2004")
    if test_topic_nb > 0:
        (robust04,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=robust04
        )

    return EvaluationsCollection(
        robust04=Evaluations(robust04, MEASURES),
    )

@lru_cache
def LoTTE_tests(test_topic_nb: int) -> EvaluationsCollection:
    """ LoTTE Search dataset """
    lotte_writing = prepare_dataset("irds.lotte.writing.test.search")
    if test_topic_nb > 0:
        (lotte_writing,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=lotte_writing
        )

    lotte_recreation = prepare_dataset("irds.lotte.recreation.test.search")
    if test_topic_nb > 0:
        (lotte_recreation,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=lotte_recreation
        )

    lotte_science = prepare_dataset("irds.lotte.science.test.search")
    if test_topic_nb > 0:
        (lotte_science,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=lotte_science
        )

    lotte_technology = prepare_dataset("irds.lotte.technology.test.search")
    if test_topic_nb > 0:
        (lotte_technology,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=lotte_technology
        )

    lotte_lifestyle = prepare_dataset("irds.lotte.lifestyle.test.search")
    if test_topic_nb > 0:
        (lotte_lifestyle,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=lotte_lifestyle
        )
    
    return EvaluationsCollection(
        lotte_writing=Evaluations(lotte_writing, MEASURES),
        lotte_recreation=Evaluations(lotte_recreation, MEASURES),
        lotte_science=Evaluations(lotte_science, MEASURES),
        lotte_technology=Evaluations(lotte_technology, MEASURES),
        lotte_lifestyle=Evaluations(lotte_lifestyle, MEASURES),
    )

@lru_cache
def paper_tests(test_topic_nb: int, include_OOD: bool = True, check_docs: bool = True) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to include in the paper.
    As of now, this list includes all of BEIR (minus the 5 datasets not publicly available) 
    + Robust04
    + LoTTE (Search)
    + the 2 TREC-DL 19 and 20 datasets, i.e.:
    - MS Marco v1 (dev set)
    - TREC DL 2019
    - TREC DL 2020
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
    v1_dev = prepare_collection("irds.msmarco-passage.dev.small")
    dl19 = prepare_dataset("irds.msmarco-passage.trec-dl-2019.judged")
    dl20 = prepare_dataset("irds.msmarco-passage.trec-dl-2020.judged")

    if test_topic_nb > 0:
        (v1_dev,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=v1_dev
        )

        (dl19,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=dl19
        )

        (dl20,) = RandomFold.folds(
            seed=0, sizes=[test_topic_nb], dataset=dl20
        )

    # Out of domain - BEIR (optional)        
    if include_OOD:
        beir = BEIR_tests(test_topic_nb)
        robust04 = Robust04_test(test_topic_nb)
        lotte = LoTTE_tests(test_topic_nb)
    else:
        # Empty collection
        beir = EvaluationsCollection()
        robust04 = EvaluationsCollection()
        lotte = EvaluationsCollection()

    paper_tests =  EvaluationsCollection(
        msmarco_dev=Evaluations(v1_dev, MEASURES),
        trec2019=Evaluations(dl19, MEASURES),
        trec2020=Evaluations(dl20, MEASURES),
        **beir.collection,
        **robust04.collection,
        **lotte.collection,
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(paper_tests)
    return paper_tests


def build_tests(
    cfg: Evaluation,
    check_docs: bool = True,
) -> EvaluationsCollection:
    """Build the tests to use for evaluation during training or at the end of it.
    :param cfg: Configuration for the evaluation
    :param check_docs: Whether to check that documents are accessible (triggers downloads if needed)
    :returns: The evaluations collection to use
    """
    
    if cfg.all_datasets or cfg.in_domain_only:
        return paper_tests(
            cfg.test_max_topics, 
            include_OOD = not cfg.in_domain_only,
            check_docs=check_docs,
        )
    else:
        return minified_tests(cfg.test_max_topics, check_docs=check_docs)