"""Utility functions for training and result processing."""

import logging
import yaml
from pathlib import Path
from attrs import asdict
import numpy as np
import pandas as pd
from jinja2 import Template

from experimaestro.annotations import tags as get_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch.trainers import LossTrainer
from xpmir.letor.trainers.batchwise import BatchwiseTrainer
from xpmir.letor.trainers.pairwise import PairwiseTrainer
from xpm_torch.losses.batchwise import SoftmaxCrossEntropy
from xpm_torch.losses.pairwise import HingeLoss, PointwiseCrossEntropyLoss

from xpmir.datasets.samplers import (
    msmarco_colbertv2_annotated,
    msmarco_rankdistillm_colbert_top50,
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_efficient_sampler,
)
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.distillation.listwise import (
    ADR_MSE,
    DistillRankNetLoss,
    DistillationListwiseTrainer,
    ListwiseSoftmaxCrossEntropy,
    ListwiseBCE,
    ListwiseHingeLoss,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)

from xpm_torch.huggingface import TorchHFHub
from configuration import Losses, CE_FineTuning

logger = logging.getLogger(__name__)


def get_task_by_tags(tasks: list, tags: dict):
    """Return the first task in tasks that has all its tags matching the given tags."""

    for task in tasks:
        task_tags = get_tags(task)
        if not task_tags:
            continue
        # Check if all given tags are present and match in the task's tags
        if all(str(task_tags.get(tag)) == str(value) for tag, value in tags.items()):
            return task
    return None


def build_trainer(cfg: CE_FineTuning) -> LossTrainer:
    """
    Builds a trainer based on the configuration's loss function.

    The trainer is responsible for the training loop, including sampling and loss calculation.
    Depending on the loss type, it returns one of:
    - `PairwiseTrainer` for BCE and Hinge loss.
    - `DistillationPairwiseTrainer` for MarginMSE.
    - `DistillationListwiseTrainer` for RankDistiLLM, DistillRankNET, and ADR_MSE.
    - `BatchwiseTrainer` for InfoNCE with in-batch negatives.

    Args:
        cfg: The fine-tuning configuration. It uses `cfg.learner.loss` to determine
            the loss function and `cfg.learner.optimization.batch_size` for the batch size.

    Returns:
        LossTrainer: A configured trainer instance from `xpm_torch` or `xpmir`.

    Raises:
        ValueError: If `cfg.learner.loss` is not a valid member of the `Losses` enum.
        NotImplementedError: If the specified loss is valid but its trainer construction
            is not implemented.
    """
    try:
        loss_member = Losses(cfg.learner.loss)
    except ValueError:
        raise ValueError(
            f"Unknown loss function: {cfg.learner.loss}. Accepted values are: {[e.value for e in Losses]}"
        )

    ### Pointwise losses
    if loss_member is Losses.BCE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=PointwiseCrossEntropyLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise losses ###
    elif loss_member is Losses.hingeLoss:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=HingeLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise distillation losses ###
    elif loss_member is Losses.marginMSE:
        return DistillationPairwiseTrainer.C(
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    ### Listwise losses ###
    elif loss_member is Losses.infoNCE_RankDistiLLM:
        passages_per_query = 8
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = batch_size // passages_per_query
            logger.warning(
                f"normalized batch size to {batch_size} to get {batch_size * passages_per_query} docs per batch"
            )
        else:
            logger.warning(
                f"Not normalizing docs per batch, {passages_per_query} docs x {batch_size} = {batch_size * passages_per_query} docs per batch"
            )

        return DistillationListwiseTrainer.C(
            sampler=msmarco_colbertv2_annotated(passages_per_query=passages_per_query),
            lossfn=ListwiseSoftmaxCrossEntropy.C(),
            batch_size=batch_size,
        )

    ### BCE and Hinge loss with ColBERT negatives ###
    elif loss_member in (Losses.BCE_RankDistiLLM, Losses.hingeLoss_RankDistiLLM):
        passages_per_query = 8
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = batch_size // passages_per_query
            logger.warning(
                f"normalized batch size to {batch_size} to get {batch_size * passages_per_query} docs per batch"
            )
        else:
            logger.warning(
                f"Not normalizing docs per batch, {passages_per_query} docs x {batch_size} = {batch_size * passages_per_query} docs per batch"
            )

        if loss_member is Losses.BCE_RankDistiLLM:
            loss_fn = ListwiseBCE.C()
        else:
            loss_fn = ListwiseHingeLoss.C()

        return DistillationListwiseTrainer.C(
            sampler=msmarco_colbertv2_annotated(passages_per_query=passages_per_query),
            lossfn=loss_fn,
            batch_size=batch_size,
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logger.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logger.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=ADR_MSE.C(),
        )
    ## Not using this one
    elif loss_member is Losses.infoNCE:
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = int(np.sqrt(batch_size))
            passages_per_batch = batch_size * batch_size
            logger.warning(
                f"normalized batch size to {batch_size} to get {passages_per_batch} docs per batch"
            )
        else:
            passages_per_batch = batch_size * batch_size
            logger.warning(
                f"Not normalizing docs per batch for InfoNCE, {batch_size}**2 docs = {passages_per_batch} docs per batch"
            )

        return BatchwiseTrainer.C(
            sampler=PairwiseInBatchNegativesSampler.C(
                sampler=msmarco_v1_docpairs_efficient_sampler(),
            ),
            lossfn=SoftmaxCrossEntropy.C(),
            batch_size=batch_size,
            hooks=[],
        )

    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


def save_raw_results(df: pd.DataFrame, resultspath: Path):
    """Formats and saves the raw experimental results to disk."""
    if not resultspath.exists():
        resultspath.mkdir(parents=True, exist_ok=True)

    output_file = resultspath / "raw_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Raw results saved to {output_file}")


def identify_best_models(
    df: pd.DataFrame, dataset: str, metric: str, group_by_tags: list
) -> pd.DataFrame:
    """Identifies the best model for each configuration based on a specific dataset and metric."""
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        logger.warning(f"Dataset {dataset} not found in results for model selection")
        return pd.DataFrame()

    if ("metric", metric) in df.columns:
        metric_col = [("metric", metric)]
    elif metric in df.columns:
        metric_col = [metric]
    else:
        logger.warning(f"Metric {metric} not found in columns")
        return pd.DataFrame()

    best_models_indices = subset.groupby(group_by_tags, dropna=False)[
        metric_col
    ].idxmax()

    if isinstance(best_models_indices, pd.DataFrame):
        best_models_indices = best_models_indices.iloc[:, 0]

    return subset.loc[best_models_indices]


def add_dataset_aggregations(
    df: pd.DataFrame,
    group_by_cols: list = None,
    aggregations: dict[str, list[str]] = None,
    add_mean: bool = True,
) -> pd.DataFrame:
    """Adds aggregate rows (e.g., mean across datasets) to the results dataframe."""

    # Handle the mean aggregation by recursion
    if add_mean:
        unique_datasets = sorted(df["dataset"].unique().tolist())
        if len(unique_datasets) > 1:
            aggregations = (aggregations or {}).copy()
            if "mean" not in aggregations:
                aggregations["mean"] = unique_datasets
        return add_dataset_aggregations(df, group_by_cols, aggregations, add_mean=False)

    if not aggregations:
        return df

    new_rows = []

    for agg_name, datasets in aggregations.items():
        present_datasets = df["dataset"].unique()
        missing = [ds for ds in datasets if ds not in present_datasets]
        if missing:
            logger.warning(
                f"Aggregation {agg_name} skipped because the following datasets are missing globally: {missing}"
            )
            continue

        mask = df["dataset"].isin(datasets)
        subset = df[mask]

        if group_by_cols:
            grouped = subset.groupby(group_by_cols, dropna=False)
            # Ensure to compute the means if and only if ALL datasets in the aggregation are present
            # for each specific group (model)
            counts = grouped["dataset"].nunique()
            agg = grouped.mean(numeric_only=True)
            agg = agg[counts == len(datasets)].reset_index()
        else:
            # Global mean
            if subset["dataset"].nunique() == len(datasets):
                agg = subset.mean(numeric_only=True).to_frame().T
            else:
                agg = pd.DataFrame()

        if not agg.empty:
            agg["dataset"] = agg_name
            new_rows.append(agg)

    if new_rows:
        new_rows = [row[row.columns.intersection(df.columns)] for row in new_rows]
        return pd.concat([df] + new_rows, ignore_index=True)

    return df


def format_model_results(
    model_df: pd.DataFrame, aggregations: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes results for a single model, adding aggregations and formatting for MD."""
    if isinstance(model_df.columns, pd.MultiIndex):
        flat_cols = []
        for col in model_df.columns:
            if col[0] == "metric":
                flat_cols.append(col[1])
            elif col[0] == "dataset":
                flat_cols.append("dataset")
            else:
                flat_cols.append("_".join(str(x) for x in col if x))
        model_df.columns = flat_cols

    metrics_to_show = ["Success@5", "RR@10", "nDCG@10"]
    cols_to_keep = [c for c in ["dataset"] + metrics_to_show if c in model_df.columns]

    if not cols_to_keep:
        return pd.DataFrame(), pd.DataFrame()

    results = model_df[cols_to_keep].copy()
    agg_names = list(aggregations.keys())
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    results[numeric_cols] = results[numeric_cols] * 100

    csv_results = results.copy()
    csv_results[numeric_cols] = csv_results[numeric_cols].round(2)

    md_results = results.copy()
    for col in numeric_cols:
        md_results[col] = md_results[col].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else x
        )

    for agg_name in agg_names:
        idx = md_results[md_results["dataset"] == agg_name].index
        if not idx.empty:
            md_results.loc[idx, "dataset"] = f"**{agg_name}**"
            for col in numeric_cols:
                md_results.loc[idx, col] = md_results.loc[idx, col].apply(
                    lambda x: f"**{x}**" if pd.notna(x) else x
                )

    return csv_results, md_results


def export_model(
    best_tags: dict,
    model_name: str,
    csv_results: pd.DataFrame,
    md_results: pd.DataFrame,
    learners: list,
    all_weights: list,
    best_cfg: CE_FineTuning,
    resultspath: Path,
    card_template_txt: str = None,
):
    """Exports all artifacts (weights, logs, readme, config) for a best model.
    Also saves the results to a CSV file and generates a README card using a template.

    Args:
        best_tags: The tags corresponding to the best model configuration.
        model_name: The name to use for the exported model (e.g., "Mice-lX+Y").
        csv_results: The results dataframe to save as CSV.
        md_results: The results dataframe to format in the README.
        learners: The list of learner tasks to search for the best model's task.
        all_weights: The list of all weight-saving tasks to find the best model's weights.
        best_cfg: The configuration of the best model, used for README generation.
        resultspath: The base path where the model artifacts and results should be saved.
        card_template_txt: Optional Jinja2 template string for the README card.
        aggregations: Optional dict of dataset aggregations to include in the README results.
    """
    models_path = resultspath / "models"
    best_model_path = models_path / model_name
    best_model_path.mkdir(parents=True, exist_ok=True)

    csv_results.to_csv(best_model_path / "results.csv", index=False)
    logger.info(f"Model results saved to {best_model_path / 'results.csv'}")

    best_model_learner = get_task_by_tags(learners, best_tags)
    if best_model_learner:
        symlink_path = best_model_path / "job_logs"
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(best_model_learner.jobpath)

        tb_path = best_model_learner.logpath
        if tb_path.exists():
            tb_symlink_path = best_model_path / "tensorboard_logs"
            if tb_symlink_path.exists():
                tb_symlink_path.unlink()
            tb_symlink_path.symlink_to(tb_path)

    if card_template_txt and best_cfg:
        template = Template(card_template_txt)
        card = template.render(
            base=best_cfg.base,
            k=best_cfg.retrieval.k,
            retriever=best_cfg.retriever if best_cfg.retriever else "BM25",
            model_id=model_name,
            training_data="MS MARCO Passage",
            dataset="msmarco",
            loss=best_tags.get("learner.loss") or best_tags.get("loss", ""),
            results=md_results.to_markdown(index=False),
        )
        with open(best_model_path / "README.md", "w") as f:
            f.write(card)

        with open(best_model_path / "config.yaml", "w") as f:
            yaml.dump(asdict(best_cfg.learner), f, default_flow_style=False)

    # 4. Export to HF format
    best_model_loader = get_task_by_tags(all_weights, best_tags)
    if best_model_loader:
        logger.info(f"Exporting model to HF format at {best_model_path}")
        hub = TorchHFHub(best_model_loader)
        hub.save_pretrained(best_model_path)
    else:
        logger.warning(f"Could not find model task for tags {best_tags} in all_weights")
