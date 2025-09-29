from functools import partial
import logging

from xpmir.distributed import DistributedHook
from xpmir.learning.learner import Learner
from xpmir.letor.learner import ValidationListener
import xpmir.letor.trainers.pairwise as pairwise
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.rankers.standard import BM25
from xpmir.papers.results import PaperResults
from xpmir.papers.helpers.samplers import (
    msmarco_v1_docpairs_efficient_sampler,
    msmarco_v1_tests,
    msmarco_v1_validation_dataset,
    prepare_collection,
)
from xpmir.neural.generative.hf import T5CustomOutputGenerator, LoadFromT5
from xpmir.neural.generative.cross import GenerativeCrossScorer
from monot5.configuration import MonoT5
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, RandomScorer
from xpmir.experiments.ir import ir_experiment, IRExperimentHelper

logging.basicConfig(level=logging.INFO)


def get_retrievers(cfg: MonoT5):
    """Returns retrievers


    :param cfg: The configuration
    :return: A tuple composed of (1) a retriever factory based on the base model
        (BM25) and (2)
    """
    launcher_index = cfg.indexation.launcher

    base_model = BM25.C()

    retrievers = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )  #: Anserini based retrievers

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
        device=cfg.device,
    )  #: Model-based retrievers

    return retrievers, model_based_retrievers


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: MonoT5) -> PaperResults:
    """monoBERT model"""

    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    device = cfg.device
    random = cfg.random

    documents = prepare_collection("irds.msmarco-passage.documents")
    ds_val = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    tests = msmarco_v1_tests(cfg.dev_test_size)

    # Setup indices and validation/test base retrievers
    retrievers, model_based_retrievers = get_retrievers(cfg)
    val_retrievers = partial(
        retrievers, store=documents, k=cfg.learner.validation_top_k
    )
    test_retrievers = partial(
        retrievers, store=documents, k=cfg.retrieval.k
    )  #: Test retrievers

    # Search and evaluate with a random re-ranker
    random_scorer = RandomScorer.C(random=random).tag("scorer", "random")
    tests.evaluate_retriever(
        partial(
            model_based_retrievers,
            retrievers=test_retrievers,
            scorer=random_scorer,
            device=None,
        ),
        launcher=launcher_preprocessing,
    )

    # Search and evaluate with the base model
    tests.evaluate_retriever(test_retrievers, cfg.indexation.launcher)

    # Define the different launchers

    # define the trainer for mono_t5
    trainer = pairwise.PairwiseTrainer.C(
        lossfn=pairwise.PointwiseCrossEntropyLoss.C(),
        sampler=msmarco_v1_docpairs_efficient_sampler(
            sample_rate=cfg.learner.sample_rate,
            sample_max=cfg.learner.sample_max,
            launcher=launcher_preprocessing,
        ),
        batcher=PowerAdaptativeBatcher.C(),
        batch_size=cfg.learner.optimization.batch_size,
    )
    
    # Modifies a T5 model so it only outputs 3 tokens (including pad which is used as <BOS>)
    generator = T5CustomOutputGenerator.C(
        tokens=["true", "false", "<pad>"], hf_id=cfg.base
    )

    # Creates a generative cross-scorer that computes P(token | q/d template)
    # where q/d template is by default: Document: [D] Query [Q] Relevant:
    mono_scorer = GenerativeCrossScorer.C(generator=generator, relevant_token_id=0).tag(
        "scorer", "mono_t5"
    )

    # The validation listener evaluates the full retriever
    # (retriever + scorer) and keep the best performing model
    # on the validation set
    validation = ValidationListener.C(
        id="bestval",
        dataset=ds_val,
        retriever=model_based_retrievers(
            documents,
            retrievers=val_retrievers,
            scorer=mono_scorer,
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
        trainer=trainer,
        # The model to train
        model=mono_scorer,
        # Optimization settings
        steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
        optimizers=cfg.learner.optimization.optimizer,
        max_epochs=cfg.learner.optimization.max_epochs,
        # The listeners (here, for validation)
        listeners=[validation],
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook.C(models=[generator]), True)],
    )

    # Submit job and link
    outputs = learner.submit(
        launcher=launcher_learner, init_tasks=[LoadFromT5.C(t5_model=generator)]
    )
    helper.tensorboard_service.add(learner, learner.logpath)

    # Evaluate the neural model on test collections
    for metric_name in validation.monitored():
        load_model = outputs.listeners[validation.id][metric_name]
        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=mono_scorer,
                retrievers=test_retrievers,
                device=device,
            ),
            launcher_evaluate,
            model_id=f"mono_t5-{metric_name}",
            init_tasks=[load_model],
        )

    return PaperResults(
        models={"mono_t5-RR@10": outputs.listeners[validation.id]["RR@10"]},
        evaluations=tests,
        tb_logs={"mono_t5-RR@10": learner.logpath},
    )
