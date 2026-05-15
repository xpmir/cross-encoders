"""
Standard Cross-Encoder Training Experiment.

This module provides the primary training pipeline for Cross-Encoders, supporting
first-stage retrieval (BM25, SPLADE) followed by neural re-ranking. It includes
support for multiple loss functions, multi-dataset validation, and
hyperparameter tuning via grid search.

"""

import logging
from functools import partial
import numpy as np

from experimaestro import setmeta, stop_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook

from xpmir.papers.results import PaperResults
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.neural.splade import splade_encoder_from_pretrained_hf

from retrievers import splade_retriever, bm25_retriever
from validations import ValidationSet
from configuration import CE_FineTuning, generate_grid
from tests import build_tests
from format import loss_names, backbone_names_lower

from training_utils import build_trainer, process_experiment_results

logging.basicConfig(level=logging.INFO)


def get_ce_name(model_tags: dict) -> str:
    """Creates the HF id from tags using formatting conventions."""
    loss = model_tags.get("loss", "")
    base = model_tags.get("base", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"
    base = backbone_names_lower.get(base, base).replace("/", "-")
    return f"cross-encoder-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: CE_FineTuning) -> PaperResults:
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    tests = build_tests(cfg.evaluation, launcher=launcher_preprocessing)

    # cache the indexes
    learners = []
    all_weights = []

    def run_one_config(
        helper: LearningExperimentHelper,
        cfg: CE_FineTuning,
        grid_search_id: str,
        cfg_tags: dict,
    ):
        """Main process for Cross-encoder training"""

        if cfg.retriever:
            # We don't use BM25, but a given sparse retriever
            retriever_tag = cfg.retriever
            splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
                cfg.retriever
            )

            # Caches the Splade index task for a document collection
            val_retrievers_factory = partial(
                splade_retriever,
                cfg,
                retriever_tag,
                splade_encoder,
                launcher_index=launcher_index,
                init_tasks=retriever_init_tasks,
                topk=cfg.learner.validation_top_k,
                in_memory=True,
            )

            test_retrievers_factory = partial(
                splade_retriever,
                cfg,
                retriever_tag,
                splade_encoder,
                launcher_index=launcher_index,
                topk=cfg.retrieval.k,
                init_tasks=retriever_init_tasks,
            )
        else:
            retriever_tag = "bm25"
            retriever_init_tasks = []  # no init task for BM25

            val_retrievers_factory = partial(
                bm25_retriever,
                cfg,
                retriever_tag,
                launcher_index=launcher_index,
                topk=cfg.learner.validation_top_k,
            )

            test_retrievers_factory = partial(
                bm25_retriever,
                cfg,
                retriever_tag,
                launcher_index=launcher_index,
                topk=cfg.retrieval.k,
            )

        # evaluate base retrievers alone and precompute runs for faster evaluation
        logging.info(f"Precomputing first stage runs for {retriever_tag}")
        test_runs = tests.evaluate_retriever(
            test_retrievers_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
        )
        test_run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_tag, test_runs
        )

        ### Validation ###
        validation_set = ValidationSet.load(cfg, launcher_preprocessing)
        val_tests = validation_set.to_evaluations()

        # Evaluate First stage on validation and store the topk
        # this enables faster validations during training
        val_runs = val_tests.evaluate_retriever(
            val_retrievers_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
        )
        val_run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_tag, val_runs
        )

        ### TRAINING CROSS ENCODER

        ce_trainer: LossTrainer = build_trainer(cfg)

        # Build the model
        scorer_model, scorer_hf_init_tasks = hf_cross_scorer(
            hf_id=cfg.base,
            max_length=cfg.max_length,
            max_query_length=cfg.max_query_length,
            max_doc_length=cfg.max_doc_length,
        )

        for k, v in cfg_tags.items():
            scorer_model.tag(k, v)
        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validations, tracked_validations = validation_set.build_listeners(
                scorer_model,
                val_run_retriever_factory,
                retriever_tag,
            )

            hooks = [setmeta(GradientLogHook.C(), True)]

            if cfg.learner.max_grad_norm > 0:
                gradient_clipping_hook = GradientClippingHook.C(
                    max_norm=cfg.learner.max_grad_norm
                )
                hooks.append(gradient_clipping_hook)

            # The learner trains the model
            learner = Learner.C(
                # Misc settings
                random=random,
                trainer=ce_trainer,  # How to train the model
                model=scorer_model,  # The model to train
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=stop_tags(validations),  # don't grab tags for validation
                # The hook used for evaluation
                hooks=hooks,
                # fabric settings
                fabric_config=cfg.learner.fabric.get_config(),
            )
            learners.append(learner)

            # Submit job and link
            outputs = learner.submit(
                launcher=launcher_learner,
                init_tasks=scorer_hf_init_tasks,
            )
            # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
            helper.tensorboard_service.add(learner, learner.logpath)

            # Evaluate each model on test collections
            for name, tracked_validation in tracked_validations.items():
                logging.info(f"evaluating from validation: {name}")
                for metric_name in tracked_validation.monitored():
                    load_model = (
                        outputs.listeners[tracked_validation.id][metric_name]
                        .tag("validation", name)
                        .tag("scorer", grid_search_id)
                        .tag("seed", seed)
                        .tag("first_stage", retriever_tag)
                    )
                    all_weights.append(load_model)
                    tests.evaluate_retriever(
                        partial(
                            scorer_retriever,
                            scorer=scorer_model,
                            retrievers=test_run_retriever_factory,
                            batch_size=cfg.retrieval.batch_size,
                        ),
                        launcher_evaluate,
                        model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                        init_tasks=[load_model],
                        with_run=cfg.save_runs,
                    )

    all_configs, all_tags = generate_grid(cfg)

    # Simplify tags keys
    new_all_tags = []
    for cfg_tags in all_tags:
        simple_tags = {}
        for k, v in cfg_tags.items():
            simple_key = k.split(".")[-1]
            if simple_key in simple_tags:
                simple_tags[k] = v
            else:
                simple_tags[simple_key] = v
        new_all_tags.append(simple_tags)
    all_tags = new_all_tags

    config_map = {}

    for config, cfg_tags in zip(all_configs, all_tags):
        # Update cfg_tags with base if not present
        if "base" not in cfg_tags:
            cfg_tags["base"] = config.base

        # just run the config
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        config_map[frozenset(cfg_tags.items())] = config
        logging.info(
            f"Running config with tags:\n- {'\n- '.join(f'{k}: {v}' for k, v in cfg_tags.items())}"
        )
        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()
    # Process Results shared
    process_experiment_results(
        tests, all_tags, config_map, cfg, helper, learners, all_weights, get_ce_name
    )
