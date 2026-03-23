"""One experiment to rule them all, merge all possible configurations here"""

import logging
import shutil
import yaml
from functools import partial
from pathlib import Path
from typing import Any
from attrs import asdict
import numpy as np
import pandas as pd
from jinja2 import Template

from experimaestro import setmeta, stop_tags
from experimaestro.annotations import tags as get_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.configuration import FabricConfiguration
from xpm_torch.losses.batchwise import SoftmaxCrossEntropy
from xpm_torch.losses.pairwise import HingeLoss, PointwiseCrossEntropyLoss
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.trainers.batchwise import BatchwiseTrainer
from xpm_torch.trainers.pairwise import PairwiseTrainer

from xpmir.papers import configuration
from xpmir.papers.results import PaperResults
from xpmir.papers.helpers.samplers import (
    msmarco_colbertv2_annotated,
    msmarco_rankdistillm_colbert_top50,
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_efficient_sampler,
)
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever
from xpmir.rankers.standard import BM25, Model
from xpmir.rankers import Documents, Retriever, scorer_retriever
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.distillation.listwise import (
    ADR_MSE,
    DistillRankNetLoss,
    DistillationListwiseTrainer,
    ListwiseSoftmaxCrossEntropy,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.evaluation import Evaluations, EvaluationsCollection

from retrievers import MultiRunRetrieverFactory


from configuration import Losses, CE_FineTuning, Validation, generate_grid
from tests import build_tests, CE_MEASURES
from format import dataframe_to_latex, aggregation_hf, loss_names, backbone_names_lower
from validations import nano_msmarco_validation_datasets, nanobeir_validation_datasets
from index_utils import get_splade_index

logging.basicConfig(level=logging.INFO)


def get_task_by_tags(tasks: list, tags: dict):
    """Return the first task in tasks that has all the given tags."""
    for task in tasks:
        task_tags = get_tags(task)
        logging.debug(f"Checking task with tags {task_tags} against {tags}")
        if all(str(task_tags.get(tag)) == str(value) for tag, value in tags.items()):
            return task
    return None


def build_trainer(cfg: CE_FineTuning) -> LossTrainer:
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
            # batcher=PowerAdaptativeBatcher.C(),
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
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise distillation losses ###
    elif loss_member is Losses.marginMSE:
        # define the trainer for monomlm
        return DistillationPairwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    ### Listwise losses ###
    elif loss_member is Losses.infoNCE_RankDistiLLM:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        # Use the listwise distillation trainer for listwise-style losses.
        # Swap to a DistillationListwiseTrainer and a listwise distillation loss.
        passages_per_query = 8
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = batch_size // passages_per_query
            logging.warning(
                f"normalized batch size to {batch_size} to get {batch_size * passages_per_query} docs per batch"
            )
        else:
            logging.warning(
                f"Not normalizing docs per batch, {passages_per_query} docs x {batch_size} = {batch_size * passages_per_query} docs per batch"
            )

        return DistillationListwiseTrainer.C(
            sampler=msmarco_colbertv2_annotated(passages_per_query=passages_per_query),
            lossfn=ListwiseSoftmaxCrossEntropy.C(),
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logging.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logging.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=ADR_MSE.C(),
        )
    ## Not using this one
    elif loss_member is Losses.infoNCE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        # Use the listwise distillation trainer for listwise-style losses.
        # Swap to a DistillationListwiseTrainer and a listwise distillation loss.
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = int(np.sqrt(batch_size))
            passages_per_batch = batch_size * batch_size
            logging.warning(
                f"normalized batch size to {batch_size} to get {passages_per_batch} docs per batch"
            )
        else:
            passages_per_batch = batch_size * batch_size
            logging.warning(
                f"Not normalizing docs per batch for InfoNCE, {batch_size}**2 docs = {passages_per_batch} docs per batch"
            )

        return BatchwiseTrainer.C(
            sampler=PairwiseInBatchNegativesSampler.C(
                sampler=msmarco_v1_docpairs_efficient_sampler(),
            ),
            lossfn=SoftmaxCrossEntropy.C(),
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
            hooks=[],
        )

    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


def bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index,
    topk: int = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""
    return anserini.AnseriniRetriever.C(
        k=topk or cfg.retrieval.k,
        model=BM25.C(),
        index=anserini.index_builder(launcher=launcher_index)(documents),
        store=documents,
    ).tag("first_stage", name)


def splade_retriever(
    cfg: CE_FineTuning,
    name: str,
    encoder: Model,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    in_memory: bool = False,
    **kwargs,
) -> Retriever.C:
    """Factory for Splade Retriever, given the current configuration"""

    return SparseRetriever.C(
        index=get_splade_index(
            documents,
            splade_encoder=encoder,
            indexation_cfg=cfg.indexation,
            launcher_index=launcher_index,
            init_tasks=init_tasks,
        ),
        topk=topk or cfg.retrieval.k,
        batchsize=1,
        encoder=encoder,
        in_memory=in_memory,
    ).tag("first_stage", name)


@configuration()
class ValidationSet:
    cfg: CE_FineTuning
    items: list[tuple[str, Any, Any]]  # (name, dataset, documents)

    @classmethod
    def load(cls, cfg: CE_FineTuning, launcher):
        items = []
        if cfg.learner.validation == Validation.MSMARCO.value:
            ds_val, docs = nano_msmarco_validation_datasets(
                cfg.validation, launcher=launcher
            )
            items.append(("msmarco", ds_val, docs))
        elif cfg.learner.validation in [
            Validation.NanoBEIR.value,
            Validation.ALL.value,
        ]:
            validations, documents = nanobeir_validation_datasets(
                cfg.validation, launcher=launcher
            )
            for name in validations:
                items.append((name, validations[name], documents[name]))
        return cls(cfg=cfg, items=items)

    def to_evaluations(self) -> EvaluationsCollection:
        """Returns an EvaluationsCollection for the validation datasets"""
        evals = {}
        for name, ds, _ in self.items:
            # Use standard measures for validation evaluations
            evals[name] = Evaluations(ds, measures=CE_MEASURES)
        return EvaluationsCollection(**evals)

    def build_listeners(
        self,
        scorer_model,
        val_retrievers_factory,
        retriever_tag,
    ) -> tuple[list[ValidationListener], dict[str, ValidationListener]]:
        """Build the validation Listeners based of the logic from Validation enum"""
        listeners = []
        msmarco_validation = None

        for name, ds, docs in self.items:
            # build the listener
            retriever = scorer_retriever(
                documents=docs,
                retrievers=val_retrievers_factory,
                scorer=scorer_model,
                batch_size=self.cfg.retrieval.batch_size,
            ).tag("first_stage", retriever_tag)

            listener = ValidationListener.C(
                id=f"bestval_zs_{name}"
                if len(self.items) > 1
                else "bestval",  # Maintain ID compatibility
                dataset=ds,
                retriever=stop_tags(retriever),  # remove dependency
                validation_interval=self.cfg.learner.validation_interval,
                metrics={"nDCG": True, "RR@10": False},
            )
            listeners.append(listener)
            if name == "msmarco":
                msmarco_validation = listener

        tracked_validations = {}

        if self.cfg.learner.validation in [
            Validation.NanoBEIR.value,
            Validation.ALL.value,
        ]:
            aggregator = AggregatorValidationListener.C(
                listeners=listeners,
                id="aggregated_validation",
                validation_interval=self.cfg.learner.validation_interval,
                metrics={"nDCG": True, "RR@10": False},
            )
            listeners.append(aggregator)
            tracked_validations["nano-beir"] = aggregator

        if self.cfg.learner.validation in [
            Validation.MSMARCO.value,
            Validation.ALL.value,
        ]:
            tracked_validations["msmarco"] = msmarco_validation

        return listeners, tracked_validations


def get_name_from_tags(model_tags: dict) -> str:
    """Creates the HF id from tags using formatting conventions."""
    loss = model_tags.get("learner.loss")
    base = model_tags.get("base")
    # try to get prettier name
    loss = loss_names.get(loss, loss).replace("/", "-")
    base = backbone_names_lower.get(base, base).replace("/", "-")
    return f"cross-encoder-{base}-{loss}"


def save_raw_results(df: pd.DataFrame, resultspath: Path):
    """Formats and saves the raw experimental results to disk."""
    # save results
    if not resultspath.exists():
        resultspath.mkdir(parents=True, exist_ok=True)

    output_file = resultspath / "raw_results.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Raw results saved to {output_file}")


def identify_best_models(
    df: pd.DataFrame, dataset: str, metric: str, group_by_tags: list
) -> pd.DataFrame:
    """Identifies the best model for each configuration based on a specific dataset and metric."""
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        logging.warning(f"Dataset {dataset} not found in results for model selection")
        return pd.DataFrame()

    # For each group (e.g. model configuration), find the row with the max metric
    # We use a list to avoid "ValueError: Cannot subset columns with a tuple" in some pandas versions
    metric_col = [("metric", metric)]
    best_models_indices = subset.groupby(group_by_tags, dropna=False)[
        metric_col
    ].idxmax()

    # If metric_col was a list, idxmax returns a DataFrame, we take the first column
    if isinstance(best_models_indices, pd.DataFrame):
        best_models_indices = best_models_indices.iloc[:, 0]

    return subset.loc[best_models_indices]


def add_dataset_aggregations(
    df: pd.DataFrame,
    group_by_cols: list,
    aggregations: dict[str, list[str]] = None,
    add_mean: bool = True,
) -> pd.DataFrame:
    """Adds aggregate rows (e.g., mean across datasets) to the results dataframe."""
    new_rows = []

    def get_agg(mask, name):
        subset = df[mask] if mask is not None else df
        if group_by_cols:
            agg = (
                subset.groupby(group_by_cols, dropna=False)
                .mean(numeric_only=True)
                .reset_index()
            )
        else:
            agg = subset.mean(numeric_only=True).to_frame().T
        agg["dataset"] = name
        return agg

    if aggregations:
        for agg_name, datasets in aggregations.items():
            # Check if all required datasets are present
            present_datasets = df["dataset"].unique()
            missing = [ds for ds in datasets if ds not in present_datasets]
            if not missing:
                mask = df["dataset"].isin(datasets)
                new_rows.append(get_agg(mask, agg_name))
            else:
                logging.warning(
                    f"Aggregation {agg_name} skipped because the following datasets are missing: {missing}"
                )

    if add_mean and df["dataset"].nunique() > 1:
        new_rows.append(get_agg(None, "mean"))

    if new_rows:
        # Filter columns to match df and avoid extra columns
        new_rows = [row[row.columns.intersection(df.columns)] for row in new_rows]
        return pd.concat([df] + new_rows, ignore_index=True)

    return df


def format_model_results(
    model_df: pd.DataFrame, aggregations: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes results for a single model, adding aggregations and formatting for MD."""
    # Flatten multi-index columns
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

    metrics_to_show = ["RR@10", "nDCG@10"]
    cols_to_keep = [c for c in ["dataset"] + metrics_to_show if c in model_df.columns]

    if not cols_to_keep:
        return pd.DataFrame(), pd.DataFrame()

    results = model_df[cols_to_keep].copy()

    # Identify which rows are aggregations for bolding later
    agg_names = list(aggregations.keys())

    # Format numeric columns: * 100
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    results[numeric_cols] = results[numeric_cols] * 100

    # Create versions for CSV (rounded) and Markdown (bolded)
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


def export_model_artifacts(
    best_tags: dict,
    scorer_tagspath: str,
    csv_results: pd.DataFrame,
    md_results: pd.DataFrame,
    learners: list,
    all_weights: list,
    best_cfg: Any,
    resultspath: Path,
    card_template_txt: str = None,
    aggregations: dict[str, list[str]] = None,
):
    """Exports all artifacts (weights, logs, readme, config) for a best model."""
    model_tags = {}
    for s in best_tags["scorer"].split("_"):
        try:
            k, v = s.split("=")
            model_tags[k] = v
        except ValueError:
            logging.warning(f"Unexpected tag format '{s}' in scorer tags")
    logging.warning(f"got tags {model_tags}")

    model_name = get_name_from_tags(model_tags)
    models_path = resultspath / "models"
    best_model_path = models_path / model_name
    best_model_path.mkdir(parents=True, exist_ok=True)

    # 1. Save results
    csv_results.to_csv(best_model_path / "results.csv", index=False)
    logging.info(f"Model results saved to {best_model_path / 'results.csv'}")

    # 2. Link logs
    best_model_learner = get_task_by_tags(learners, best_tags)
    if best_model_learner:
        # Job logs
        symlink_path = best_model_path / "job_logs"
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(best_model_learner.jobpath)

        # TensorBoard
        tb_path = best_model_learner.logpath
        if tb_path.exists():
            tb_symlink_path = best_model_path / "tensorboard_logs"
            if tb_symlink_path.exists():
                tb_symlink_path.unlink()
            tb_symlink_path.symlink_to(tb_path)

    # 3. Weights
    best_model_val = get_task_by_tags(all_weights, best_tags)
    if best_model_val and best_model_val:
        weights_path = best_model_val.loader.path
        if weights_path.name.endswith(".pth"):
            shutil.copy(weights_path, best_model_path / "model_weights.pt")
        else:
            logging.warning(f"Model weights is not a file: {weights_path}")

    # 4. Model Card & Config
    if card_template_txt and best_cfg:
        template = Template(card_template_txt)
        card = template.render(
            base=best_cfg.base,
            k=best_cfg.retrieval.k,
            retriever=best_cfg.retriever if best_cfg.retriever else "BM25",
            model_id=model_name,
            training_data="MS MARCO Passage",
            dataset="msmarco",
            loss=model_tags.get("learner.loss"),
            results=md_results.to_markdown(index=False),
        )
        with open(best_model_path / "README.md", "w") as f:
            f.write(card)

        with open(best_model_path / "config.yaml", "w") as f:
            yaml.dump(asdict(best_cfg.learner), f, default_flow_style=False)


def compute_aggregated_results(
    df: pd.DataFrame,
    metric_cols: list,
    group_by_tags: list,
    resultspath: Path,
    aggregations: dict[str, list[str]] = None,
):
    """Computes final grouped results across all experiments and saves to CSV/LaTeX."""
    df_grouped = (
        df.groupby(["dataset"] + group_by_tags, dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    df_grouped = df_grouped.sort_index(axis=1)

    output_file = resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)

    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=None,
    )
    with open(resultspath / "results.tex", "w") as f:
        f.write(latex_table)


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
        helper: LearningExperimentHelper, cfg: CE_FineTuning, grid_search_id: str
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
        # Build the model
        scorer_model, scorer_hf_init_tasks = hf_cross_scorer(
            hf_id=cfg.base, max_doc_length=cfg.max_doc_len
        )
        scorer_model.tag("scorer", grid_search_id)

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
                fabric_config=FabricConfiguration.C(
                    strategy=cfg.learner.strategy,
                    precision=cfg.learner.precision,
                    accelerator=cfg.learner.accelerator,
                ),
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
                    )

    all_configs, all_tags = generate_grid(cfg)
    config_map = {}

    for config, cfg_tags in zip(all_configs, all_tags):
        # just run the config
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        config_map[tagspath] = config
        logging.info(f"Running config with tags {tagspath}")
        run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    # Constants
    group_by_tags = [("tag", "first_stage"), ("tag", "scorer")]
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

    # keep only results with a scorer
    scorer_only_df = df_with_aggs[
        df_with_aggs[("tag", "scorer")].notna()
        & (df_with_aggs[("tag", "scorer")] != "")
    ]

    logging.info(scorer_only_df)

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
        for _, best_row in best_models_df.iterrows():
            # Extract tags for this best model
            best_tags = {
                tag[1]: best_row[tag] for tag in model_id_tags if tag[0] == "tag"
            }

            scorer_tagspath = best_tags["scorer"]
            logging.info(f"Best evaluated model is {best_tags}")

            # Filter the original dataframe for this specific best model (all datasets)
            mask = pd.Series(True, index=df.index)
            for tag in model_id_tags:
                mask &= df[tag].astype(str) == str(best_row[tag])

            best_model_df = df[mask].copy()
            best_models_list.append(best_model_df)

            # Format and Export artifacts
            csv_results, md_results = format_model_results(
                best_model_df, aggregations=aggregation_hf
            )

            export_model_artifacts(
                best_tags=best_tags,
                scorer_tagspath=scorer_tagspath,
                csv_results=csv_results,
                md_results=md_results,
                learners=learners,
                all_weights=all_weights,
                best_cfg=config_map.get(scorer_tagspath),
                resultspath=helper.xp.resultspath,
                card_template_txt=card_template_txt,
                aggregations=aggregation_hf,
            )

        if best_models_list:
            best_model_df = pd.concat(best_models_list, ignore_index=True)
            best_model_df.to_csv(
                helper.xp.resultspath / "best_models_per_scorer_raw_results.csv",
                index=False,
            )

    # Final aggregation and LaTeX table generation
    compute_aggregated_results(
        df,
        metric_cols,
        group_by_tags,
        helper.xp.resultspath,
        aggregations=aggregation_hf,
    )
