"""
MICE (Minimal-interaction Cross-Encoder) Training Experiment.

This module implements the training pipeline for Cross-Encoders using various
loss functions (Pointwise, Pairwise, Listwise, Distillation). It handles
multi-dataset validation, grid search over hyperparameters, and automated
model card generation for the best performing models.

It leverages shared utilities from `training_utils`, `retrievers`, and
`validations` to maintain a concise and maintainable implementation.
"""

import logging
import shutil
from functools import partial
from typing import Optional
from pathlib import Path
from MICE.modeling.plaid_mice import InitEncoderFromMice
import numpy as np
import pandas as pd

from experimaestro import setmeta, stop_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook

from xpmir.index.plaid import PlaidIndexBuilder, PlaidRetriever
from xpmir.papers.results import PaperResults
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.text.huggingface.tokenizers import get_default_max_len
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration

from MICE.modeling.mice import mice_scorer
from retrievers import splade_retriever, bm25_retriever
from validations import ValidationSet
from configuration import generate_grid, CE_FineTuning
from tests import build_tests
from format import aggregations, dataframe_to_latex, loss_names, backbone_names_lower

from training_utils import (
    build_trainer,
    save_raw_results,
    identify_best_models,
    add_dataset_aggregations,
    format_model_results,
    export_model,
)

logging.basicConfig(level=logging.INFO)


@configuration()
class Mice_FineTuning(CE_FineTuning):
    ## MICE specific configuration
    n_contextualization_layers: int = 6
    """Number of bottom encoder layers that process query and document independently"""

    n_interaction_layers: Optional[int] = None
    """Number of top encoder layers with cross-attention. If None, use all remaining layers from the backbone."""

    bound_bottom_layers: Optional[bool] = True
    """whether to bound bottom query and document encoding layers"""

    cross_attn_first: bool = True
    """Whether to perform cross-attention before self-attention in the top layers."""

    mask_cls_to_doc: bool = True
    """Whether to mask the [CLS] token from attending to document tokens."""

    mask_query_to_cls: bool = True
    """Whether to mask query tokens from attending to the [CLS] token (using it as a sink)"""

    freeze_base: bool = False
    """Whether to freeze the bottom layers during finetuning"""

    random_top_layers: bool = False
    """Whether to initialize top layers randomly instead of copying from backbone"""

    global_cls_token: bool = False
    """Whether to add a fresh [CLS] token before the top layers."""

    compress_dim: float = 1.0
    """Factor by which to divide the hidden dimensions of the top layers"""

    save_runs: bool = False
    """Whether to save the evaluation runs in the best model folders"""


def get_name_from_tags(model_tags: dict, cfg: Mice_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    n_ctx_layers = model_tags.get(
        "n_contextualization_layers", cfg.n_contextualization_layers
    )
    n_inter_layers = model_tags.get("n_interaction_layers", cfg.n_interaction_layers)
    cross_attn_first = model_tags.get("cross_attn_first", cfg.cross_attn_first)
    vanilla = "-vanilla" if not cross_attn_first else ""

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    # try to get prettier name
    if len(loss):
        loss = f"-{loss}"

    return f"Mice-l{n_ctx_layers}+{n_inter_layers}{vanilla}-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: Mice_FineTuning) -> PaperResults:
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
        cfg: Mice_FineTuning,
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

        default_max_len = get_default_max_len(cfg.base)
        if cfg.max_length and default_max_len > cfg.max_length:
            max_len = cfg.max_length
        else:
            max_len = None
            logging.warning(
                f"No max_len provided or default max_len {default_max_len} is not greater than provided max_len {cfg.max_length}. Using default max_len {default_max_len} for scorer {cfg.base}"
            )
        # Build the model using the unified scorer factory
        mice_model, scorer_hf_init_tasks = mice_scorer(
            hf_id=cfg.base,
            n_contextualization_layers=cfg.n_contextualization_layers,
            n_interaction_layers=cfg.n_interaction_layers,
            bound_bottom_layers=cfg.bound_bottom_layers,
            mask_cls_to_doc=cfg.mask_cls_to_doc,
            mask_query_to_cls=cfg.mask_query_to_cls,
            cross_attn_first=cfg.cross_attn_first,
            freeze_base=cfg.freeze_base,
            random_top_layers=cfg.random_top_layers,
            compress_dim=cfg.compress_dim,
            global_cls_token=cfg.global_cls_token,
            pooling_method=cfg.pooling_method,
            max_length=max_len,
        )
        for k, v in cfg_tags.items():
            mice_model.tag(k, v)

        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validations, tracked_validations = validation_set.build_listeners(
                mice_model,
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
                model=mice_model,  # The model to train
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
                if not cfg.plaid.use_plaid:
                    for metric_name in tracked_validation.monitored():
                        load_model = (
                            outputs.listeners[tracked_validation.id][metric_name]
                            .tag("validation", name)
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                        )
                        for k, v in cfg_tags.items():
                            load_model.tag(k, v)

<<<<<<< HEAD
                    all_weights.append(load_model)
                    tests.evaluate_retriever(
                        partial(
                            scorer_retriever,
                            scorer=mice_model,
                            retrievers=test_run_retriever_factory,
                            batch_size=cfg.retrieval.batch_size,
                        ),
                        launcher_evaluate,
                        model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                        init_tasks=[load_model],
                        with_run=cfg.save_runs,
=======
                        all_weights.append(load_model)
                        tests.evaluate_retriever(
                            partial(
                                scorer_retriever,
                                scorer=mice_model,
                                retrievers=test_run_retriever_factory,
                                batch_size=cfg.retrieval.batch_size,
                            ),
                            launcher_evaluate,
                            model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                            init_tasks=[load_model],
                        )
                else:
                    logging.info(
                        f"Running PLAID-style evaluation from validation: {name}"
>>>>>>> dd94f2ea590930d3dfd8203d7c35b573670e43e0
                    )
                    for metric_name in tracked_validation.monitored():
                        load_model = (
                            outputs.listeners[tracked_validation.id][metric_name]
                            .tag("validation", name)
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                        )
                        for k, v in cfg_tags.items():
                            load_model.tag(k, v)
                        
                        doc_encoder = mice_model.get_document_encoder()
                        
                        # 1) Build the index and retriever for this model
                        for dataset, documents in test_run_retriever_factory.documents.items():
                            logging.info(f"Building PLAID index for dataset {dataset}")
                            plaid_index = PlaidIndexBuilder.C(
                                documents=documents.tag("dataset", dataset),
                                encoder=doc_encoder,
                                batch_size=cfg.plaid.batch_size,
                                n_bits=cfg.plaid.n_bits,
                                kmeans_niters=cfg.plaid.kmeans_niters,
                                n_samples_kmeans=cfg.plaid.n_samples_kmeans,
                                compress_only=cfg.plaid.compress_only,
                                
                            ).submit(launcher=launcher_index, init_tasks=[load_model])

                            # plaid_retriever = PlaidRetriever.C(
                            #     store=documents.tag("dataset", dataset),
                            #     index=plaid_index,
                            #     encoder=mice_model,
                            #     topk=cfg.retrieval.k,
                            #     n_ivf_probe=cfg.plaid.n_ivf_probe,
                            #     n_full_scores=cfg.plaid.n_full_scores,
                            # )

                            # 2) Run tests
                            # all_weights.append(load_model)
                            # tests.evaluate_retriever(
                            #     plaid_retriever,
                            #     launcher_evaluate,
                            #     model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                            #     init_tasks=[load_model],
                            # )

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

    # Constants
    grid_keys = set()
    for tags in all_tags:
        grid_keys.update(tags.keys())

    # Ensure unique tags for grouping
    tag_names = {"first_stage"} | grid_keys
    group_by_tags = [("tag", k) for k in sorted(list(tag_names))]
    model_id_tags = group_by_tags + [("tag", "seed")]

    # 1. Exctract data
    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    logging.info(f"Evaluated models: \n- {'\n- '.join(tests.per_model.keys())}")

    # Convert to numeric
    metric_cols = [col for col in df.columns if col[0] == "metric"]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    # Add both specific aggregations (ID, BEIR, etc.) and the global mean
    df_with_aggs = add_dataset_aggregations(
        df,
        group_by_cols=model_id_tags,
        aggregations=aggregations,
        add_mean=True,  # will add 'mean' dataset at the end
    )

    save_raw_results(df_with_aggs, helper.xp.resultspath)

    # keep only results with a base tag (all our models have it)
    # We ensure it's not NaN and not an empty string to exclude first-stage only results
    scorer_only_df = df_with_aggs[
        df_with_aggs[("tag", "base")].notna()
        & (df_with_aggs[("tag", "base")].astype(str) != "")
        & (df_with_aggs[("tag", "base")].astype(str) != "nan")
    ]

    # Read model card template
    template_path = Path(__file__).parent / "CrossEncoderCard.md"
    card_template_txt = template_path.read_text() if template_path.exists() else None

    # Identify and export best models per configuration
    best_models_df = identify_best_models(
        scorer_only_df,
        dataset="mean",
        metric="nDCG@10",
        group_by_tags=group_by_tags,
    )
    logging.info(f"df with only best models:\n{best_models_df}")

    best_models_list = []
    if not best_models_df.empty:
        # Check if model folder exists before saving models, delete if so
        models_path = helper.xp.resultspath / "models"
        if models_path.exists():
            shutil.rmtree(models_path)
            logging.info(f"Deleted existing models directory: {models_path}")

        for _, best_row in best_models_df.iterrows():
            # Extract tags for this best model
            best_tags = {
                tag[1]: best_row[tag] for tag in model_id_tags if tag[0] == "tag"
            }

            logging.info(f"Best evaluated model is {best_tags}")

            # Reconstruct the grid tags to find the original config
            best_grid_tags = {k: best_tags[k] for k in grid_keys if k in best_tags}
            best_cfg = config_map.get(frozenset(best_grid_tags.items()))
            scorer_tagspath = "_".join(
                f"{k}={v}" for k, v in sorted(best_grid_tags.items())
            )

            # Filter the original dataframe for this specific best model (all datasets)
            mask = pd.Series(True, index=df_with_aggs.index)
            for tag in model_id_tags:
                mask &= df_with_aggs[tag].astype(str) == str(best_row[tag])

            best_model_df = df_with_aggs[mask].copy()
            best_models_list.append(best_model_df)

            # Format and Export artifacts
            csv_results, md_results = format_model_results(
                best_model_df, aggregations=aggregations
            )

            model_name = get_name_from_tags(best_tags, all_configs[0])
            if (helper.xp.resultspath / "models" / model_name).exists():
                scorer_path = scorer_tagspath.replace("/", "-")
                logging.warning(
                    f"Model directory for {model_name} already exists. using {scorer_path} as model name instead to avoid overwriting."
                )
                model_name = scorer_path

            # Collect evaluation results for the best model

            export_model(
                best_tags=best_tags,
                model_name=model_name,
                csv_results=csv_results,
                md_results=md_results,
                learners=learners,
                all_weights=all_weights,
                best_cfg=best_cfg,
                resultspath=helper.xp.resultspath,
                card_template_txt=card_template_txt,
                save_runs=cfg.save_runs,
                tests=tests,
            )

        if best_models_list:
            best_model_df = pd.concat(best_models_list, ignore_index=True)
            best_model_df.to_csv(
                helper.xp.resultspath / "best_models_per_scorer_raw_results.csv",
                index=False,
            )

    # Final aggregation and LaTeX table generation
    df_grouped = (
        df_with_aggs.groupby(["dataset"] + group_by_tags, dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    df_grouped = df_grouped.sort_index(axis=1)
    logging.info(df_grouped)
    df_grouped.to_csv(helper.xp.resultspath / "results.csv", index=False)

    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=None,
    )
    with open(helper.xp.resultspath / "results.tex", "w") as f:
        f.write(latex_table)
    logging.info(f"Saved aggregated results to {helper.xp.resultspath / 'results.csv'}")
    logging.info("Experiment completed successfully.")
