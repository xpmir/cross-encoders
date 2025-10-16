import logging
from functools import partial

from monoMLM.configuration import MonoMLM
from experimaestro import setmeta, tag
from experimaestro.launcherfinder import find_launcher

import xpmir.interfaces.anserini as anserini
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.distributed import DistributedHook
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.learner import Learner
from xpmir.letor.learner import ValidationListener
from xpmir.neural.cross import CrossScorer
from xpmir.papers.helpers.samplers import (
    msmarco_v1_docpairs_efficient_sampler,
    msmarco_v1_validation_dataset, prepare_collection, MEASURES)
from xpmir.papers.results import PaperResults
from xpmir.rankers import RandomScorer, scorer_retriever
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import HFCLSEncoder, HFStringTokenizer
from xpmir.text import TokenizedTextEncoder
from datamaestro import prepare_dataset

from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection


logging.basicConfig(level=logging.INFO)


def get_retrievers(cfg: MonoMLM):
    """Returns retrievers


    :param cfg: The configuration
    :return: A tuple composed of (1) a retriever factory based on the base model
        (BM25) and (2)
    """
    launcher_index = cfg.indexation.launcher

    base_model = BM25.C().tag("retriever", "BM25")

    retrievers = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
        device=cfg.device,
    )  #: Model-based retrievers

    return retrievers, model_based_retrievers

def get_beir_14_datasets(dev_test_size: int = 0):
    """Returns the pool of queries to use for the evaluations."""

    v1_devsmall_ds = prepare_collection("irds.msmarco-passage.dev.small") # 1
    if dev_test_size > 0:
        (v1_devsmall_ds,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=v1_devsmall_ds
        )
    mini_arguana = prepare_dataset("irds.beir.arguana") # 2
    if dev_test_size > 0:
        (mini_arguana,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_arguana
        )
    mini_climate_fever = prepare_dataset("irds.beir.climate-fever") # 3
    if dev_test_size > 0:
        (mini_climate_fever,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_climate_fever
        )
    mini_fever = prepare_dataset("irds.beir.fever.test") # 4
    if dev_test_size > 0:
        (mini_fever,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_fever
        )
    mini_dbpedia = prepare_dataset("irds.beir.dbpedia-entity.test") # 5
    if dev_test_size > 0:
        (mini_dbpedia,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_dbpedia
        )
    mini_hotpotqa = prepare_dataset("irds.beir.hotpotqa.test") # 6
    if dev_test_size > 0:
        (mini_hotpotqa,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_hotpotqa
        )
    mini_quora = prepare_dataset("irds.beir.quora.test") # 7
    if dev_test_size > 0:
        (mini_quora,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_quora
        )
    mini_scidocs = prepare_dataset("irds.beir.scidocs") # 8
    if dev_test_size > 0:
        (mini_scidocs,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_scidocs
        )
    mini_scifact = prepare_dataset("irds.beir.scifact.test") # 9
    if dev_test_size > 0:
        (mini_scifact,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_scifact
        )
    touche = prepare_dataset("irds.beir.webis-touche2020.v2") # 10
    mini_trec_covid = prepare_dataset("irds.beir.trec-covid") # 11
    if dev_test_size > 0:
        (mini_trec_covid,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_trec_covid
        )
    mini_fiqa = prepare_dataset("irds.beir.fiqa.test") # 12
    if dev_test_size > 0:
        (mini_fiqa,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_fiqa
        )

    mini_nfcorpus = prepare_dataset("irds.beir.nfcorpus.test") # 13
    if dev_test_size > 0:
        (mini_nfcorpus,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_nfcorpus
        )
    mini_nq = prepare_dataset("irds.beir.nq") # 14
    if dev_test_size > 0:
        (mini_nq,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=mini_nq
        )
    return EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, MEASURES),
        arguana=Evaluations(mini_arguana, MEASURES),
        climate_fever=Evaluations(mini_climate_fever, MEASURES),
        fever=Evaluations(mini_fever, MEASURES),
        dbpedia=Evaluations(mini_dbpedia, MEASURES),
        hotpotqa=Evaluations(mini_hotpotqa, MEASURES),
        quora=Evaluations(mini_quora, MEASURES),
        scidocs=Evaluations(mini_scidocs, MEASURES),
        trec_covid=Evaluations(mini_trec_covid, MEASURES),
        scifact=Evaluations(mini_scifact, MEASURES),
        touche=Evaluations(touche, MEASURES),
        fiqa=Evaluations(mini_fiqa, MEASURES),
        nfcorpus=Evaluations(mini_nfcorpus, MEASURES),
        nq=Evaluations(mini_nq, MEASURES),
    )

@ir_experiment()
def run(helper: IRExperimentHelper, cfg: MonoMLM) -> PaperResults:
    """monoMLM model training"""

    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    device = cfg.device
    random = cfg.random

    documents = prepare_collection("irds.msmarco-passage.documents")
    ds_val = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    tests = get_beir_14_datasets(cfg.dev_test_size)

    # Setup indices and validation/test base retrievers
    retrievers, model_based_retrievers = get_retrievers(cfg)
    val_retrievers = partial(
        retrievers, store=documents, k=cfg.learner.validation_top_k
    )

    random_scorer = RandomScorer.C(random=random).tag("scorer", "random")
    for evaluation in tests.evaluations(model_id=f"{cfg.id}-base"):
        # IR documents
        documents = evaluation.dataset.documents.tag("data", evaluation.key)

        # Build the retriever for those
        test_retrievers = partial(
            retrievers, store=documents, k=cfg.retrieval.k
        )  #: Test retrievers

        # Search and evaluate with the base model
        evaluation.evaluate(test_retrievers, launcher=cfg.indexation.launcher)

        # Search and evaluate with a random re-ranker
        evaluation.evaluate(
            partial(
                model_based_retrievers,
                retrievers=test_retrievers,
                scorer=random_scorer,
                device=None,
            ),
            launcher=launcher_preprocessing,
        )

    # define the trainer for monomlm
    monomlm_trainer = pairwise.PairwiseTrainer.C(
        lossfn=pairwise.PointwiseCrossEntropyLoss.C(),
        sampler=msmarco_v1_docpairs_efficient_sampler(
            sample_rate=cfg.learner.sample_rate,
            sample_max=cfg.learner.sample_max,
            launcher=launcher_preprocessing,
        ),
        batcher=PowerAdaptativeBatcher.C(),
        batch_size=cfg.learner.optimization.batch_size,
    )

    monomlm_scorer: CrossScorer = CrossScorer.C(
        encoder=TokenizedTextEncoder.C(
            tokenizer=HFStringTokenizer.from_pretrained_id(cfg.base),
            encoder=HFCLSEncoder.from_pretrained_id(cfg.base),
        )
    ).tag("scorer", cfg.id)

    # The validation listener evaluates the full retriever
    # (retriever + scorer) and keep the best performing model
    # on the validation set
    validation = ValidationListener.C(
        id="bestval",
        dataset=ds_val,
        retriever=model_based_retrievers(
            documents,
            retrievers=val_retrievers,
            scorer=monomlm_scorer,
            device=device,
        ),
        validation_interval=cfg.learner.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG": False},
    )

    # The learner trains the model
    learner = Learner.C(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=monomlm_trainer,
        # The model to train
        model=monomlm_scorer,
        # Optimization settings
        steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
        optimizers=cfg.learner.optimization.optimizer,
        max_epochs=cfg.learner.optimization.max_epochs,
        # The listeners (here, for validation)
        listeners=[validation],
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook.C(models=[monomlm_scorer]), True)],
    )

    # Submit job and link
    outputs = learner.submit(launcher=launcher_learner)
    helper.tensorboard_service.add(learner, learner.logpath)

    # Evaluate the neural model on test collections
    for metric_name in validation.monitored():
        load_model = outputs.listeners[validation.id][metric_name]
        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=monomlm_scorer,
                retrievers=test_retrievers,
                device=device,
            ),
            launcher_evaluate,
            model_id=f"{cfg.id}-{metric_name}",
            init_tasks=[load_model],
        )
        for evaluation in tests.evaluations(model_id=f"{cfg.id}-{metric_name}"):
            # IR documents
            documents = evaluation.dataset.documents.tag("data", evaluation.key)

            # Build the retriever for those
            test_retrievers = partial(
                retrievers, store=documents, k=cfg.retrieval.k
            )  #: Test retrievers

            # Search and evaluate with a random re-ranker
            evaluation.evaluate(
                retriever=partial(
                    model_based_retrievers,
                    scorer=monomlm_scorer,
                    retrievers=test_retrievers,
                    device=device,
                ),
                launcher=launcher_evaluate,
                init_tasks=[load_model],
            )

    return PaperResults(
        models={"{cfg.id}-RR@10": outputs.listeners[validation.id]["RR@10"]},
        evaluations=tests,
        tb_logs={"{cfg.id}-RR@10": learner.logpath},
    )
