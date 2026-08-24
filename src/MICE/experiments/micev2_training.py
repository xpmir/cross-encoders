"""
MICE V2 (ModernBERT-based) Training Experiment.
Implements training pipeline for MICE V2 models.
"""

import logging
from functools import partial
from typing import Optional, Callable, List, Tuple
import numpy as np

from experimaestro import setmeta, stop_tags, LightweightTask, RunMode
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch.huggingface import prepare_hf_model

from xpmir.papers.results import PaperResults
from xpmir.rankers import AbstractModuleScorer, scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.text.huggingface.tokenizers import get_default_max_len
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration

from retrievers import splade_retriever, bm25_retriever, MicePlaidRetrieverFactory
from validations import ValidationSet
from experimaestro.experiments.grid import generate_grid, GridSearch
from configuration import CE_FineTuning, CheckpointSelection
from tests import build_tests
from format import (
    loss_names,
    backbone_names_lower,
)

from training_utils import (
    build_trainer,
    process_experiment_results,
)

from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    UtilityRegularizer,
    InitMICEV2FromCheckpoint,
)
from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class MiceV2_FineTuning(CE_FineTuning):
    ## MICE V2 specific configuration
    n_contextualization_layers: GridSearch[int] = 4
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: GridSearch[Optional[int]] = None
    """Number of bottom encoder layers for the document. If None, use n_contextualization_layers."""

    n_interaction_layers: GridSearch[Optional[int]] = None
    """Number of top encoder layers with cross-attention. If None, use all remaining layers from backbone."""

    bound_bottom_layers: GridSearch[bool] = True
    """Whether to bound bottom query and document encoding layers"""

    attn_implementation: GridSearch[Optional[str]] = None
    """Attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    utility_pred: GridSearch[bool] = False
    """Whether to enable 2-layer MLP soft-gating utility predictor on document tokens"""

    lambda_reg: GridSearch[float] = 0.0
    """Regularization weight for the Utility predictor"""

    phase2_training: GridSearch[bool] = False
    """Whether to run Phase 2 hard token pruning fine-tuning after Phase 1 soft gating"""

    hard_prune_threshold: GridSearch[float] = 0.5
    """Utility threshold below which document tokens are hard dropped in Phase 2"""

    normalize_doc_embeds: GridSearch[bool] = False
    """Whether to L2-normalize document hidden states before utility prediction and upper interaction layers"""

    phase2_max_epochs: GridSearch[int] = 3
    """Number of training epochs for Phase 2 top layer fine-tuning"""


def build_mice_v2(
    cfg: MiceV2_FineTuning,
    phase_2: bool = False,
) -> Tuple[AbstractModuleScorer, List[LightweightTask]]:
    """Build the MICE V2 model using the modernbert_xpm factory."""
    default_max_len = get_default_max_len(cfg.base)
    if cfg.max_length and default_max_len > cfg.max_length:
        max_len = cfg.max_length
    else:
        max_len = default_max_len

    attn_impl = cfg.attn_implementation or cfg.pref_attn_implementation

    if phase_2:
        logging.info(
            f"[Phase 2] Building MICE V2 model for hard pruning fine-tuning (threshold={cfg.hard_prune_threshold})"
        )
        return mice_v2_scorer(
            hf_id=cfg.base,
            n_contextualization_layers=cfg.n_contextualization_layers,
            n_docs_ctx_layers=cfg.n_docs_ctx_layers,
            n_interaction_layers=cfg.n_interaction_layers,
            bound_bottom_layers=cfg.bound_bottom_layers,
            attn_implementation=attn_impl,
            utility_pred=True,
            hard_pruning=True,
            hard_prune_threshold=cfg.hard_prune_threshold,
            normalize_doc_embeds=cfg.normalize_doc_embeds,
            max_length=max_len,
        )
    else:
        logging.info(
            f"[Phase 1] Building MICE V2 model for soft gating training (utility_pred={cfg.utility_pred})"
        )
        return mice_v2_scorer(
            hf_id=cfg.base,
            n_contextualization_layers=cfg.n_contextualization_layers,
            n_docs_ctx_layers=cfg.n_docs_ctx_layers,
            n_interaction_layers=cfg.n_interaction_layers,
            bound_bottom_layers=cfg.bound_bottom_layers,
            attn_implementation=attn_impl,
            utility_pred=cfg.utility_pred,
            normalize_doc_embeds=cfg.normalize_doc_embeds,
            max_length=max_len,
        )


def get_name_from_tags(model_tags: dict, cfg: MiceV2_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    n_ctx_layers = model_tags.get(
        "n_contextualization_layers", cfg.n_contextualization_layers
    )
    n_docs_layers = model_tags.get("n_docs_ctx_layers", cfg.n_docs_ctx_layers)
    doc_suffix = f"-d{n_docs_layers}" if n_docs_layers is not None else ""

    attn_impl = model_tags.get("attn_implementation", cfg.attn_implementation)
    attn_suffix = f"-{attn_impl}" if attn_impl else ""

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"MiceV2-l{n_ctx_layers}{doc_suffix}{attn_suffix}-{base}{loss}"


def TrainEvaluateMiceV2(
    helper: LearningExperimentHelper,
    cfg: CE_FineTuning,
    get_model_name_fn: Callable,
) -> PaperResults:
    """Generic training and evaluation experiment for late interaction models."""
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    long_reqs = cfg.retrieval.long_requirements
    long_launcher = find_launcher(long_reqs) if long_reqs else None

    tests = build_tests(
        cfg.evaluation,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        blocking=cfg.preprocessing.blocking_download,
    )

    # cache the indexes
    learners = []
    all_weights = []

    def run_one_config(
        helper: LearningExperimentHelper,
        cfg: CE_FineTuning,
        grid_search_id: str,
        cfg_tags: dict,
    ):
        """Main process for one configuration of the late interaction model."""

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
            fabric_config=cfg.evaluation.fabric.get_config(),
        )
        test_run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_tag, test_runs
        )

        ### Validation ###
        validation_set = ValidationSet.load(cfg, launcher_preprocessing)
        val_tests = validation_set.to_evaluations()
        val_runs = val_tests.evaluate_retriever(
            val_retrievers_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
            fabric_config=cfg.evaluation.fabric.get_config(),
        )

        val_run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_tag, val_runs
        )

        ### TRAINING CROSS ENCODER

        ce_trainer: LossTrainer = build_trainer(cfg)

        # Base hooks shared by all training phases across all seed repetitions
        learner_hooks = [setmeta(GradientLogHook.C(), True)]
        if cfg.learner.max_grad_norm > 0:
            gradient_clipping_hook = GradientClippingHook.C(
                max_norm=cfg.learner.max_grad_norm
            )
            learner_hooks.append(gradient_clipping_hook)

        # Phase 1 hooks: base hooks + UtilityRegularizer (if soft gating utility prediction is active)

        if cfg.utility_pred and cfg.lambda_reg > 0.0:
            histogram_interval = (
                cfg.learner.validation_interval
                * cfg.learner.optimization.steps_per_epoch
            )
            ce_trainer.hooks.append(
                UtilityRegularizer.C(
                    lambda_reg=cfg.lambda_reg,
                    histogram_interval=histogram_interval,
                    sparsity_threshold=cfg.utility_sparsity_threshold,
                )
            )

        # Build the model using the provided factory
        model, model_init_tasks = build_mice_v2(cfg)

        # tag the model with the configuration tags for monitoring and logging
        for k, v in cfg_tags.items():
            model.tag(k, v)

        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validations, tracked_validations = validation_set.build_listeners(
                model,
                val_run_retriever_factory,
                retriever_tag,
            )

            optimizers = cfg.learner.optimization.optimizer
            if cfg.phase2_training:
                total_epochs = (
                    cfg.learner.optimization.max_epochs + cfg.phase2_max_epochs
                )
                total_steps = total_epochs * cfg.learner.optimization.steps_per_epoch
                optimizers = cfg.learner.optimization.get_optimizers_with_total_steps(
                    total_steps
                )

            # The learner trains the model
            learner = Learner.C(
                # Misc settings
                random=random,
                trainer=ce_trainer,  # How to train the model
                model=model,  # The model to train
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=optimizers,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=stop_tags(validations),  # don't grab tags for validation
                # The hook used for evaluation
                hooks=learner_hooks,
                # fabric settings
                fabric_config=cfg.learner.fabric.get_config(),
            )
            learners.append(learner)

            # Submit job and link
            outputs = learner.submit(
                launcher=launcher_learner,
                init_tasks=model_init_tasks,
            )
            # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
            if helper.xp.run_mode == RunMode.NORMAL:
                helper.tensorboard_service.add(learner, learner.logpath)

            if cfg.phase2_training:
                total_epochs = (
                    cfg.learner.optimization.max_epochs + cfg.phase2_max_epochs
                )

                logging.info(
                    f"[Phase 2] Launching Hard Pruning Fine-Tuning (threshold={cfg.hard_prune_threshold}, max_epochs={cfg.phase2_max_epochs}, total_epochs={total_epochs})"
                )
                # Build the Phase 2 model for hard pruning fine-tuning
                model_phase2, _ = build_mice_v2(cfg, phase_2=True)

                for k, v in cfg_tags.items():
                    model_phase2.tag(k, v)
                model_phase2.tag("phase", 2)

                validations_p2, tracked_validations_p2 = validation_set.build_listeners(
                    model_phase2,
                    val_run_retriever_factory,
                    retriever_tag,
                )

                learner_p2 = Learner.C(
                    random=random,
                    trainer=build_trainer(
                        cfg
                    ),  # Build a new trainer for phase 2, as it may have different hooks (e.g., no utility regularizer)
                    model=model_phase2,
                    steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                    optimizers=optimizers,
                    max_epochs=cfg.phase2_max_epochs,
                    checkpoint_interval=cfg.learner.checkpoint_interval,
                    listeners=stop_tags(validations_p2),
                    hooks=learner_hooks,
                    fabric_config=cfg.learner.fabric.get_config(),
                )
                learners.append(learner_p2)

                p2_init_task = InitMICEV2FromCheckpoint.C(
                    scorer=model_phase2,
                    checkpoint_loader=outputs.learned_model,  # use outputs from phase 1 as init for phase 2
                )

                outputs_p2 = learner_p2.submit(
                    launcher=launcher_learner,
                    init_tasks=[p2_init_task],
                )
                if helper.xp.run_mode == RunMode.NORMAL:
                    helper.tensorboard_service.add(learner_p2, learner_p2.logpath)

                eval_outputs = outputs_p2
                eval_tracked_validations = tracked_validations_p2
                eval_model = model_phase2
            else:
                eval_outputs = outputs
                eval_tracked_validations = tracked_validations
                eval_model = model

            # Build list of models to evaluate: (model_id, load_model, label)
            models_to_evaluate = []
            checkpoint_mode = cfg.evaluation.get_checkpoint_mode()

            # 1. Best validation checkpoints
            if checkpoint_mode in (CheckpointSelection.VAL, CheckpointSelection.BOTH):
                for name, tracked_validation in eval_tracked_validations.items():
                    logging.info(f"Adding validation model for evaluation: {name}")
                    for metric_name in tracked_validation.monitored():
                        load_model = (
                            eval_outputs.listeners[tracked_validation.id][metric_name]
                            .tag("checkpoint", name)
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                        )
                        for k, v in cfg_tags.items():
                            load_model.tag(k, v)
                        model_id = f"{grid_search_id}-{name}-{metric_name}-{seed}"
                        models_to_evaluate.append((model_id, load_model, name))

            # 2. Last checkpoint
            if checkpoint_mode in (CheckpointSelection.LAST, CheckpointSelection.BOTH):
                logging.info("Adding last checkpoint for evaluation")
                last_model = (
                    eval_outputs.learned_model.tag("checkpoint", "last")
                    .tag("seed", seed)
                    .tag("first_stage", retriever_tag)
                )
                for k, v in cfg_tags.items():
                    last_model.tag(k, v)
                model_id = f"{grid_search_id}-last-{seed}"
                models_to_evaluate.append((model_id, last_model, "last"))

            # Shared evaluation execution loop (Standard & PLAID)
            for model_id, load_model, label in models_to_evaluate:
                if cfg.plaid.use_plaid:
                    load_model.tag("plaid_retriever", False)

                all_weights.append(load_model)
                tests.evaluate_retriever(
                    partial(
                        scorer_retriever,
                        scorer=eval_model,
                        retrievers=test_run_retriever_factory,
                        batch_size=cfg.retrieval.batch_size,
                    ),
                    launcher_evaluate,
                    model_id=model_id,
                    init_tasks=[load_model],
                    with_run=cfg.save_runs,
                    fabric_config=cfg.evaluation.fabric.get_config(),
                )

                if cfg.plaid.use_plaid:
                    load_model.tag("plaid_retriever", True)
                    logging.info(f"Running PLAID-style evaluation for: {label}")
                    # 1) Build the Retriever factory for first stage + model + Plaid
                    mice_plaid_factory = MicePlaidRetrieverFactory(
                        mice_model=eval_model,
                        first_stage_factory=test_run_retriever_factory,
                        topk=cfg.retrieval.k,
                        batchsize=cfg.retrieval.batch_size,
                    )
                    # 2) Build the plaid indexes for all datasets in the collection
                    mice_plaid_factory.build_indexes(
                        evaluations=tests,
                        cfg=cfg.plaid,
                        batch_size=cfg.retrieval.batch_size,
                        seed=seed,
                        launcher_index=launcher_index,
                        init_tasks=[load_model],
                    )

                    # 2) Run tests with the factory
                    all_weights.append(load_model)
                    tests.evaluate_retriever(
                        retriever=mice_plaid_factory,
                        launcher=launcher_evaluate,
                        model_id=f"{model_id}-Plaid",
                        init_tasks=[load_model],
                        with_run=cfg.save_runs,
                        fabric_config=cfg.evaluation.fabric.get_config(),
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

        prepare_hf_model(config.base)

        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    if helper.xp.run_mode == RunMode.NORMAL:
        process_experiment_results(
            tests=tests,
            all_tags=all_tags,
            config_map=config_map,
            cfg=cfg,
            helper=helper,
            learners=learners,
            all_weights=all_weights,
            get_name_fn=partial(get_model_name_fn, cfg=all_configs[0]),
        )
    else:
        logging.info("Dry-run mode: skipping results processing")


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: MiceV2_FineTuning) -> PaperResults:
    """Main entry point for MICE V2 training experiment."""
    return TrainEvaluateMiceV2(
        helper=helper,
        cfg=cfg,
        get_model_name_fn=get_name_from_tags,
    )
