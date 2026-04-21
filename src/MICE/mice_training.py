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
from pathlib import Path
import numpy as np
import pandas as pd

from experimaestro import setmeta, stop_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook

from xpmir.papers.results import PaperResults
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.neural.splade import splade_encoder_from_pretrained_hf

from MICE.modeling.mice import mice_scorer
from retrievers import splade_retriever, bm25_retriever
from validations import ValidationSet
from configuration import Mice_FineTuning, generate_grid
from tests import build_tests
from format import aggregation_hf, dataframe_to_latex, loss_names, backbone_names_lower

from training_utils import (
    build_trainer,
    save_raw_results,
    identify_best_models,
    add_dataset_aggregations,
    format_model_results,
    export_model,
)

logging.basicConfig(level=logging.INFO)


# TODO format as Mice-lX+Y
def get_name_from_tags(model_tags: dict, cfg: Mice_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    n_ctx_layers = model_tags.get(
        "n_contextualization_layers", cfg.n_contextualization_layers
    )
    n_inter_layers = model_tags.get("n_interaction_layers", cfg.n_interaction_layers)

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    # try to get prettier name
    if len(loss):
        loss = f"-{loss}"

    return f"Mice-l{n_ctx_layers}+{n_inter_layers}-{base}{loss}"


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

        # Build the model using the unified scorer factory
        mice_model, scorer_hf_init_tasks = mice_scorer(
            hf_id=cfg.base,
            n_contextualization_layers=cfg.n_contextualization_layers,
            n_interaction_layers=cfg.n_interaction_layers,
            mask_cls_to_doc=cfg.mask_cls_to_doc,
            mask_query_to_cls=cfg.mask_query_to_cls,
            freeze_base=cfg.freeze_base,
            random_top_layers=cfg.random_top_layers,
            compress_dim=cfg.compress_dim,
            global_cls_token=cfg.global_cls_token,
            pooling_method=cfg.pooling_method,
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
                for metric_name in tracked_validation.monitored():
                    load_model = (
                        outputs.listeners[tracked_validation.id][metric_name]
                        .tag("validation", name)
                        .tag("seed", seed)
                        .tag("first_stage", retriever_tag)
                    )
                    for k, v in cfg_tags.items():
                        load_model.tag(k, v)

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
        aggregations=aggregation_hf,
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
                best_model_df, aggregations=aggregation_hf
            )

            model_name = get_name_from_tags(best_tags, all_configs[0])
            if (helper.xp.resultspath / "models" / model_name).exists():
                scorer_path = scorer_tagspath.replace("/", "-")
                logging.warning(
                    f"Model directory for {model_name} already exists. using {scorer_path} as model name instead to avoid overwriting."
                )
                model_name = scorer_path

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
