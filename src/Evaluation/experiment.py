from typing import List
import logging
from attrs import Factory
from functools import partial
import pandas as pd

from experimaestro.launcherfinder import find_launcher

from xpm_torch.utils.huggingface import prepare_hf_model
from datamaestro_ir.data import Documents

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.index.sparse import SparseRetriever
from xpmir.neural.splade import splade_encoder_from_pretrained_hf, SpladeTextEncoder
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
from retrievers import MultiRunRetrieverFactory

logging.basicConfig(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)
    preprocessing: Preprocessing = Factory(Preprocessing)

    scorers_hf_id: List[str] = []

    retrievers_hf_id: List[str] = [""]

    evaluation: Evaluation = Factory(Evaluation)

    retrievers_only: bool = False
    """If true, only evaluate first-stage retrievers without cross-encoders"""


### Contexual Retriever Factory
def bm25_retriever(
    cfg: BaselinesConfig, name: str, documents: Documents, launcher_index
) -> Retriever.C:
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


def splade_retriever(
    cfg: BaselinesConfig,
    name: str,
    encoder: SpladeTextEncoder,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    **kwargs,
) -> Retriever.C:
    """Factory for Splade Retriever, given the current configuration"""

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


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: BaselinesConfig) -> PaperResults:
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    # Built tests collections depending on config
    if cfg.evaluation.all_datasets:
        tests = paper_tests(
            cfg.evaluation.test_max_topics,
            include_OOD=not cfg.evaluation.in_domain_only,
            retrievers_only=cfg.retrievers_only,
            launcher=launcher_preprocessing,
        )
    else:
        tests = minified_tests(
            cfg.evaluation.test_max_topics,
            include_OOD=not cfg.evaluation.in_domain_only,
            retrievers_only=cfg.retrievers_only,
            launcher=launcher_preprocessing,
        )

    # Built Retrievers - list of splade models or just bm25
    all_retrievers = []
    if len(cfg.retrievers_hf_id) > 0:
        for retriever_hf_id in cfg.retrievers_hf_id:
            if not retriever_hf_id:
                continue

            logging.info(f"Instantiating retriever {retriever_hf_id} ")

            splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
                retriever_hf_id
            )

            all_retrievers.append(
                (
                    retriever_hf_id,
                    partial(
                        splade_retriever,
                        cfg,
                        retriever_hf_id,
                        splade_encoder,
                        launcher_index=launcher_index,
                        init_tasks=retriever_init_tasks,
                    ),
                    retriever_init_tasks,
                )
            )
    else:
        # add bm25 by default
        all_retrievers.append(
            (
                "bm25",
                partial(bm25_retriever, cfg, "bm25", launcher_index=launcher_index),
                [],
            )
        )

    # Evaluate First stage retrievers only and store the results to reuse them with a second stage cross-encoder
    for retriever_name, retriever_factory, retriever_init_tasks in all_retrievers:
        # Eval First stage only
        eval_results = tests.evaluate_retriever(
            retriever_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
        )

        # Create a MultiRunRetrieverFactory storing results from first stage
        run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_name, eval_results
        )

        logging.info(
            f"First stage only evaluation done for {retriever_name} on datasets {list(run_retriever_factory.runs.keys())}"
        )
        logging.info(f"Evaluating model-based retrievers {cfg.scorers_hf_id}")

        # Eval With cross-encoder
        if cfg.retrievers_only:
            if len(cfg.scorers_hf_id) > 0:
                logging.warning(
                    "Scorers specified in config but retrievers_only is True. Skipping cross-encoder evaluation."
                )
            continue

        for scorer_hf_id in cfg.scorers_hf_id:
            # Build the cross encoder
            prepare_hf_model(scorer_hf_id)
            scorer, ce_init_tasks = hf_cross_scorer(hf_id=scorer_hf_id)
            scorer.tag("scorer", scorer_hf_id)

            # Build the two stage retriever with the run_retriever
            two_stage_retriever_factory = partial(
                scorer_retriever,
                batch_size=cfg.retrieval.batch_size,
                #   batcher=PowerAdaptativeBatcher.C(),
                scorer=scorer,
                retrievers=run_retriever_factory,
            )

            # Evaluate
            tests.evaluate_retriever(
                two_stage_retriever_factory,
                launcher=launcher_evaluate,
                init_tasks=ce_init_tasks,
            )

    # Wait for all tasks to complete
    helper.xp.wait()

    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    measures = ["AP", "RR@10", "nDCG@10"] if not cfg.retrievers_only else ["R@1000"]
    metric_cols = [("metric", measure) for measure in measures]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            (
                ["dataset", ("tag", "first_stage"), ("tag", "scorer")]
                if ("tag", "scorer") in df.columns
                else ["dataset", ("tag", "first_stage")]
            ),
            dropna=False,
        )[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    # Generate and save LaTeX table
    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=None,
        metric_col="nDCG@10" if not cfg.retrievers_only else "R@1000",
    )
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")
