"""
PreTTR (Pre-calculated Token-level Task-specific Representations) Training Experiment.

This module implements the training pipeline for PreTTR Cross-Encoders.
It uses joint tokenization but prevents cross-attention in early layers
to allow for offline document representation precomputation.
"""

import logging
import shutil
from functools import partial
from pathlib import Path
from experimaestro.scheduler.transient import TransientMode
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

from MICE.experiments.baselines.prettr_mice import prettr_scorer
from retrievers import splade_retriever, bm25_retriever
from validations import ValidationSet
from configuration import generate_grid, CE_FineTuning
from tests import build_tests
from format import (
    aggregations,
    aggregation_hf,
    loss_names,
    backbone_names_lower,
)

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
class PreTTR_FineTuning(CE_FineTuning):
    ## PreTTR specific configuration
    join_layer: int = 6
    """The layer index at which full self-attention begins."""

    prettr_max_query_length: int = 32
    """The fixed offset used for offline document precomputation."""


def get_name_from_tags(model_tags: dict, cfg: PreTTR_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    join_layer = model_tags.get("join_layer", cfg.join_layer)

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"PreTTR-j{join_layer}-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: PreTTR_FineTuning) -> PaperResults:
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
        cfg: PreTTR_FineTuning,
        grid_search_id: str,
        cfg_tags: dict,
    ):
        """Main process for PreTTR training"""

        if cfg.retriever:
            # We don't use BM25, but a given sparse retriever
            retriever_tag = cfg.retriever
            splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
                cfg.retriever
            )

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
            retriever_init_tasks = []

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

        # evaluate base retrievers
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

        ### TRAINING PRETT R

        ce_trainer: LossTrainer = build_trainer(cfg)

        default_max_len = get_default_max_len(cfg.base)
        max_len = (
            cfg.max_length
            if cfg.max_length and default_max_len > cfg.max_length
            else default_max_len
        )

        # Build the PreTTR model
        prettr_model, scorer_hf_init_tasks = prettr_scorer(
            hf_id=cfg.base,
            join_layer=cfg.join_layer,
            max_length=max_len,
            prettr_max_query_length=cfg.prettr_max_query_length,
        )
        for k, v in cfg_tags.items():
            prettr_model.tag(k, v)

        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

            validations, tracked_validations = validation_set.build_listeners(
                prettr_model,
                val_run_retriever_factory,
                retriever_tag,
            )

            hooks = [setmeta(GradientLogHook.C(), True)]

            if cfg.learner.max_grad_norm > 0:
                gradient_clipping_hook = GradientClippingHook.C(
                    max_norm=cfg.learner.max_grad_norm
                )
                hooks.append(gradient_clipping_hook)

            learner = Learner.C(
                random=random,
                trainer=ce_trainer,
                model=prettr_model,
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                listeners=stop_tags(validations),
                hooks=hooks,
                fabric_config=cfg.learner.fabric.get_config(),
            )
            learners.append(learner)

            outputs = learner.submit(
                launcher=launcher_learner,
                init_tasks=scorer_hf_init_tasks,
            )
            helper.tensorboard_service.add(learner, learner.logpath)

            # Evaluate
            for name, tracked_validation in tracked_validations.items():
                logging.info(f"evaluating from validation: {name}")
                if not cfg.plaid.use_plaid:
                    for metric_name in tracked_validation.monitored():
                        load_model = (
                            outputs.listeners[tracked_validation.id][metric_name]
                            .tag("validation", name)
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                            .tag("plaid_retriever", False)
                        )
                        for k, v in cfg_tags.items():
                            load_model.tag(k, v)

                        all_weights.append(load_model)
                        tests.evaluate_retriever(
                            partial(
                                scorer_retriever,
                                scorer=prettr_model,
                                retrievers=test_run_retriever_factory,
                                batch_size=cfg.retrieval.batch_size,
                            ),
                            launcher_evaluate,
                            model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                            init_tasks=[load_model],
                            with_run=cfg.save_runs,
                        )
                else:
                    logging.info(
                        f"Running PLAID-style evaluation from validation: {name}"
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

                        doc_encoder = prettr_model.get_document_encoder()

                        for (
                            dataset,
                            documents,
                        ) in test_run_retriever_factory.documents.items():
                            logging.info(f"Building PLAID index for dataset {dataset}")
                            plaid_index = (
                                PlaidIndexBuilder.C(
                                    documents=documents,
                                    encoder=doc_encoder,
                                    warmup_docs=cfg.plaid.warmup_docs,
                                    batch_size=cfg.indexation.batch_size,
                                    fast_plaid_batch_size=cfg.plaid.batch_size,
                                    n_bits=cfg.plaid.n_bits,
                                    kmeans_niters=cfg.plaid.kmeans_niters,
                                    n_samples_kmeans=cfg.plaid.n_samples_kmeans,
                                    seed=seed,
                                    compress_only=cfg.plaid.compress_only,
                                )
                                .tag("dataset", dataset)
                                .submit(
                                    launcher=launcher_index,
                                    init_tasks=[load_model],
                                    transient=TransientMode.REMOVE,
                                )
                            )

                            plaid_retriever_inst = PlaidRetriever.C(
                                store=documents,
                                index=stop_tags(plaid_index),
                                encoder=prettr_model,
                                topk=cfg.retrieval.k,
                                n_ivf_probe=cfg.plaid.n_ivf_probe,
                                n_full_scores=cfg.plaid.n_full_scores,
                            ).tag("plaid_retriever", True)

                            all_weights.append(load_model)
                            tests.evaluate_retriever(
                                plaid_retriever_inst,
                                launcher_evaluate,
                                model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}-{dataset}",
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
        if "base" not in cfg_tags:
            cfg_tags["base"] = config.base
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        config_map[frozenset(cfg_tags.items())] = config
        logging.info(
            f"Running config with tags:\n- {'\n- '.join(f'{k}: {v}' for k, v in cfg_tags.items())}"
        )
        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    helper.xp.wait()

    # Post-processing results
    grid_keys = set()
    for tags in all_tags:
        grid_keys.update(tags.keys())

    tag_names = {"first_stage"} | grid_keys
    group_by_tags = sorted(list(tag_names))
    model_id_tags = sorted(group_by_tags + ["seed"])

    df = tests.to_dataframe()
    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    metric_cols = [col for col in df.columns if col[0] == "metric"]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    cols_to_drop = [col for col in df.columns if "index_doc" in str(col).lower()]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df_with_aggs = add_dataset_aggregations(
        df,
        group_by_cols=model_id_tags,
        aggregations=aggregations,
        add_mean=True,
    )

    save_raw_results(df_with_aggs, helper.xp.resultspath)

    scorer_only_df = df_with_aggs[
        df_with_aggs["base"].notna()
        & (df_with_aggs["base"].astype(str) != "")
        & (df_with_aggs["base"].astype(str) != "nan")
    ]

    template_path = Path(__file__).parent / "CrossEncoderCard.md"
    card_template_txt = template_path.read_text() if template_path.exists() else None

    best_models_df = identify_best_models(
        scorer_only_df,
        dataset="mean",
        metric="nDCG@10",
        group_by_tags=group_by_tags,
    )

    best_models_list = []
    if not best_models_df.empty and cfg.export_trained_models:
        models_path = helper.xp.resultspath / "models"
        if models_path.exists():
            shutil.rmtree(models_path)

        for _, best_row in best_models_df.iterrows():
            best_tags = {tag: best_row[tag] for tag in model_id_tags}
            best_grid_tags = {k: best_tags[k] for k in grid_keys if k in best_tags}
            best_cfg = config_map.get(frozenset(best_grid_tags.items()))

            mask = pd.Series(True, index=df_with_aggs.index)
            for tag in model_id_tags:
                mask &= df_with_aggs[tag].astype(str) == str(best_row[tag])

            best_model_df = df_with_aggs[mask].copy()
            best_models_list.append(best_model_df)

            csv_results, md_results = format_model_results(
                best_model_df, aggregations=aggregation_hf
            )

            export_model(
                best_tags=best_tags,
                model_name=get_name_from_tags(best_tags, all_configs[0]),
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

    metric_names = [col[1] for col in metric_cols]
    df_grouped = (
        df_with_aggs.groupby(["dataset"] + group_by_tags, dropna=False)[metric_names]
        .agg(["mean", "var"])
        .reset_index()
    )

    df_grouped.to_csv(helper.xp.resultspath / "results.csv", index=False)
    logging.info("Experiment completed successfully.")
