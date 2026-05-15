"""Utility functions for training and result processing."""

import logging
import yaml
import shutil
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
from xpmir.evaluation import EvaluationsCollection
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
    ListwiseHingeLoss,
    ListwiseBCE,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)

from xpm_torch.huggingface import TorchHFHub
from configuration import Losses, CE_FineTuning
from samplers import msmarco_rankdistillm_sampled_colbert50

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

    ### Listwise losses with ColBERT negatives ###
    elif loss_member in (
        Losses.BCE_Colbertv2Neg,
        Losses.hingeLoss_Colbertv2Neg,
        Losses.infoNCE_Colbertv2Neg,
        Losses.infoNCE_RankDistiLLM,
    ):
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

        if loss_member in (Losses.infoNCE_Colbertv2Neg, Losses.infoNCE_RankDistiLLM):
            loss_fn = ListwiseSoftmaxCrossEntropy.C()
        elif loss_member is Losses.hingeLoss_Colbertv2Neg:
            loss_fn = ListwiseHingeLoss.C()
        else:
            loss_fn = ListwiseBCE.C()

        if loss_member is Losses.infoNCE_RankDistiLLM:
            sampler = msmarco_rankdistillm_sampled_colbert50(
                passages_per_query=passages_per_query
            )
        else:
            sampler = msmarco_colbertv2_annotated(passages_per_query=passages_per_query)

        return DistillationListwiseTrainer.C(
            sampler=sampler,
            lossfn=loss_fn,
            batch_size=batch_size,
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logger.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 50 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logger.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 50 passages per batch)."
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
    save_runs: bool = False,
    tests: EvaluationsCollection = None,
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
        save_runs: Whether to save the evaluation runs in the best model folders.
        tests: Optional EvaluationsCollection containing the evals
    """
    models_path = resultspath / "models"
    best_model_path = models_path / model_name
    if best_model_path.exists():
        scorer_path = best_tags.replace("/", "-")
        logging.warning(
            f"Model directory for {model_name} already exists. using {scorer_path} as model name instead to avoid overwriting."
        )

    best_model_path.mkdir(parents=True, exist_ok=True)

    csv_results.to_csv(best_model_path / "results.csv", index=False)
    logger.info(f"Model results saved to {best_model_path / 'results.csv'}")

    learner_tags = {
        k: v for k, v in best_tags.items() if k not in ["validation", "metric"]
    }
    best_model_learner = get_task_by_tags(learners, learner_tags)
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

    def get_runs_per_tags(model_tags: dict):
        runs = {}
        detailed = {}
        for dataset, evals in tests.collection.items():
            for eval_tags, evaluate in evals.per_tags.items():
                # Check if all given tags are present and match in the task's tags
                if all(
                    str(eval_tags.get(tag)) == str(value)
                    for tag, value in model_tags.items()
                ):
                    job_path = Path(evaluate.results).parent
                    run_path = job_path / "run.txt"
                    if run_path.exists():
                        runs[dataset] = run_path
                    else:
                        logger.warning(
                            f"Didn't found run.txt in {job_path} for {dataset}"
                        )
                    detailed_path = job_path / "detailed.dat"
                    if detailed_path.exists():
                        detailed[dataset] = detailed_path
                    else:
                        logger.warning(
                            f"Didn't found detailed.dat in {job_path} for {dataset}"
                        )
        return runs, detailed

    # 5. Export runs
    if save_runs:
        if not tests:
            logger.error("save_runs is True but no tests provided, skipping...")
        runs, detailed = get_runs_per_tags(learner_tags)
        if not runs:
            logging.warning(f"not runs retrieved for model with tags {learner_tags}")
            return
        runs_dir = best_model_path / "evals"
        runs_dir.mkdir(parents=True, exist_ok=True)
        # for dataset, runpath in runs.items():
        #     shutil.copy(runpath, runs_dir / f"run_{dataset}.txt")
        for dataset, dpath in detailed.items():
            shutil.copy(dpath, runs_dir / f"detailed_{dataset}.dat")
        logger.info(f"Copied {len(list(runs.keys()))} runs to {runs_dir}")


def check_detailed_results(detailed_path: Path, metric_name: str = "nDCG@10"):
    """
    Check if there are any zero scores for a given metric in a detailed.dat file.

    The format of detailed.dat is:
    {:25s} {:10s} {:.4f}
    (Metric Name) (Query ID) (Value)
    """
    if not detailed_path.exists():
        logger.error(f"Detailed results file not found at {detailed_path}")
        return

    zeros_count = 0
    total_count = 0

    try:
        with detailed_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue

                # The format is fixed width or space separated
                parts = line.split()
                if len(parts) < 3:
                    continue

                # Metric name is the first part (can be multiple parts if not careful,
                # but split() handles spaces)
                current_metric = parts[0]
                value = float(parts[-1])

                if current_metric == metric_name:
                    total_count += 1
                    if value == 0.0:
                        zeros_count += 1

        if total_count == 0:
            logger.warning(
                f"No results found for metric '{metric_name}' in {detailed_path}"
            )
        elif zeros_count > 0:
            percentage = (zeros_count / total_count) * 100
            logger.warning(
                f"FOUND {zeros_count}/{total_count} ({percentage:.1f}%) ZERO SCORES "
                f"for metric '{metric_name}' in {detailed_path}. "
                "This might indicate a synchronization issue in Multi-GPU inference."
            )
        else:
            logger.info(
                f"All {total_count} queries for '{metric_name}' have non-zero scores in {detailed_path}"
            )

    except Exception as e:
        logger.error(f"Error reading detailed results: {e}")
