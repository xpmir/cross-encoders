from typing import List, Optional
import logging
from attrs import Factory
from functools import partial
import pandas as pd
from pathlib import Path

from experimaestro.launcherfinder import find_launcher
from experimaestro import PathSerializationLWTask
from datamaestro_ir.data import Documents

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.index.sparse import SparseRetriever
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration
from xpmir.papers.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.rankers.standard import BM25
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, Retriever

from format import dataframe_to_latex
from tests import minified_tests, paper_tests

from configuration import Retrieval, Indexation, Preprocessing, Evaluation
from index_utils import get_splade_index

logging.basicConfig(level=logging.INFO)


@configuration()
class LocalEvalsConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)
    preprocessing: Preprocessing = Factory(Preprocessing)

    model_root: str = ""
    """Root path for all models"""

    base: str = ""
    """Identifier for the base model"""

    max_doc_len: Optional[int] = None
    """max len for scorer, default to 0 = max len of the model"""

    models: List[str] = []
    """List of directory names for the models to evaluate"""

    retriever_hf_id: str = ""
    """HF ID for the first-stage retriever (e.g. SPLADE). If empty, uses BM25."""

    evaluation: Evaluation = Factory(Evaluation)


class CELoader(PathSerializationLWTask):
    def execute(self):
        """Loads the model from disk using the given serialization path"""
        import torch

        # first initialize model structure (empty init)
        self.value.initialize()
        # then load state dict
        logging.info("Loading model from disk: %s", self.path)
        data = torch.load(self.path)

        self.value.encoder.load_state_dict(data)


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: LocalEvalsConfig) -> PaperResults:
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    # Build the tests
    if cfg.evaluation.all_datasets:
        tests = paper_tests(
            cfg.evaluation.test_max_topics,
            include_OOD=not cfg.evaluation.in_domain_only,
            launcher=launcher_preprocessing,
        )
    else:
        tests = minified_tests(
            cfg.evaluation.test_max_topics,
            include_OOD=not cfg.evaluation.in_domain_only,
            launcher=launcher_preprocessing,
        )

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
    )

    ### BM25 Retriever
    def bm25_retriever(name, documents: Documents) -> Retriever.C:
        return (
            anserini.AnseriniRetriever.C(
                k=cfg.retrieval.k,
                model=BM25.C(),
                index=anserini.index_builder(launcher=launcher_index)(documents),
                store=documents,
            )
            .tag("first_stage", name)
            .tag("data", documents.id)
        )

    ### SPLADE Retriever
    def splade_retriever(
        name,
        encoder,
        documents: Documents,
        init_tasks: list = None,
    ) -> Retriever.C:
        return (
            SparseRetriever.C(
                index=get_splade_index(
                    documents,
                    splade_encoder=encoder,
                    indexation_cfg=cfg.indexation,
                    launcher_index=launcher_index,
                    init_tasks=init_tasks,
                ),
                topk=cfg.retrieval.k,
                batchsize=1,
                encoder=encoder,
                in_memory=False,
            )
            .tag("first_stage", name)
            .tag("data", documents.id)
        )

    # Determine first-stage retriever
    if cfg.retriever_hf_id:
        logging.info(f"Instantiating retriever {cfg.retriever_hf_id}")
        splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
            cfg.retriever_hf_id
        )
        retriever_factory = partial(
            splade_retriever,
            cfg.retriever_hf_id,
            splade_encoder,
            init_tasks=retriever_init_tasks,
        )
    else:
        logging.info("Using BM25 as first-stage retriever")
        retriever_factory = partial(bm25_retriever, "bm25")
        retriever_init_tasks = []

    # Identify models to evaluate
    root_path = Path(cfg.model_root)

    if not cfg.models:
        logging.warning("No models specified in the 'models' list.")

    for model_name in cfg.models:
        model_path = root_path / model_name
        weights_path = model_path / "model_weights.pt"

        if weights_path.exists():
            logging.info(f"Evaluating model: {model_name} from {weights_path}")
            # Build the model
            scorer, ce_init_tasks = hf_cross_scorer(
                hf_id=cfg.base, max_doc_length=cfg.max_doc_len
            )
            scorer.tag("scorer", model_name)

            # Load the weights
            load_task = CELoader.C(path=weights_path, value=scorer)
            ce_init_tasks.append(load_task)

            # evaluate with the underlying First stage retriever
            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer,
                    retrievers=retriever_factory,
                ),
                launcher=launcher_evaluate,
                init_tasks=retriever_init_tasks + ce_init_tasks,
            )
        elif (model_path / "config.json").exists():
            logging.info(f"Evaluating model: {model_path} (HuggingFace format)")
            # Build the model
            scorer, ce_init_tasks = hf_cross_scorer(hf_id=str(model_path))
            scorer.tag("scorer", Path(model_path).name)

            # evaluate with the underlying First stage retriever
            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer,
                    retrievers=retriever_factory,
                ),
                launcher=launcher_evaluate,
                init_tasks=retriever_init_tasks + ce_init_tasks,
            )
        else:
            logging.warning(f"No valid model found in {model_path}, skipping")

    helper.xp.wait()

    df = tests.to_dataframe()
    if df.empty:
        logging.info("No results found")
        return

    measures = ["AP", "RR@10", "nDCG@10"]
    metric_cols = [("metric", measure) for measure in measures]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    group_cols = ["dataset", ("tag", "first_stage"), ("tag", "scorer")]
    df_grouped = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)

    # Generate and save LaTeX table
    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results for Local Models",
        label="tab:local_eval_results",
        metric_col="nDCG@10",
    )
    with open(helper.xp.resultspath / "results.tex", "w") as f:
        f.write(latex_table)

    return PaperResults(results=df_grouped)
