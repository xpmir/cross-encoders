"""
Two-Stage Information Retrieval Evaluation Experiment.

This module provides the infrastructure to evaluate Information Retrieval (IR) systems,
supporting both first-stage retrievers (BM25, SPLADE) and two-stage systems
integrating Cross-Encoders for re-ranking.

The experiment workflow includes:
1. Configuration of retrieval, indexation, and preprocessing parameters.
2. Building test collections across multiple datasets.
3. First-stage evaluation: Running and caching results for base retrievers.
4. Second-stage evaluation: Applying Cross-Encoders (scorers) to the first-stage runs.
5. Result processing: Aggregating metrics (RR@10, nDCG@10, R@1000) and
   generating standardized outputs (CSV, LaTeX tables).

Usage:
    This module is designed to be invoked via experimaestro:
    `uv run experimaestro run-experiment src/Evaluation/cross-encoders.yaml`
"""

from typing import List, Optional
from attrs import Factory
from functools import partial
import pandas as pd

from experimaestro import RunMode
from experimaestro.launcherfinder import find_launcher
from experimaestro.experiments.grid import generate_grid

from xpm_torch.huggingface import prepare_hf_model
from xpm_torch.configuration import FabricConfiguration

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration
from xpmir.experiments.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.neural.sentence_transformers import st_cross_scorer, st_multivector_scorer
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory

from format import dataframe_to_latex, aggregations
from tests import build_tests
from configuration import (
    Retrieval,
    Indexation,
    Preprocessing,
    Evaluation,
    ScorerType,
    PlaidConfiguration,
)
from retrievers import splade_retriever, bm25_retriever, plaid_retriever
from training_utils import add_dataset_aggregations

import logging

from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)
    preprocessing: Preprocessing = Factory(Preprocessing)

    scorers_hf_id: List[str] = []
    scorers_st_id: List[str] = []
    scorers_st_multivector_id: List[str] = []

    attn_implementation: Optional[str] = None
    """Preferred attention implementation (e.g. 'flash_attention_2', 'sdpa')"""

    precision: Optional[str] = None
    """Precision setting for evaluation (e.g. 'bf16-mixed')"""

    max_length: Optional[int] = None
    """Maximum length for cross-encoders"""

    max_query_length: Optional[int] = None
    """Maximum query length for cross-encoders"""

    max_doc_length: Optional[int] = None
    """Maximum document length for cross-encoders"""

    retrievers_hf_id: List[str] = []

    use_plaid: bool = False
    """If true, evaluate retrievers_hf_id using PLAID multi-vector retrieval instead of SPLADE"""

    plaid: PlaidConfiguration = Factory(PlaidConfiguration)
    """Configuration for PLAID indexing and retrieval"""

    include_recall_100: bool = False
    """Whether to additionally report R@100 alongside CE_MEASURES"""

    evaluation: Evaluation = Factory(Evaluation)

    retrievers_only: bool = False
    """If true, only evaluate first-stage retrievers without cross-encoders"""


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: BaselinesConfig) -> PaperResults:
    configs, tags = generate_grid(cfg)

    if len(configs) > 1:
        raise ValueError(
            f"Evaluation script does not support grid search, but {len(configs)} "
            "configurations were generated. Please remove multiple values from your YAML."
        )

    cfg = configs[0]

    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    fabric_config = (
        cfg.evaluation.fabric.get_config()
        if hasattr(cfg.evaluation, "fabric")
        else (
            FabricConfiguration.C(precision=cfg.precision)
            if getattr(cfg, "precision", None)
            else None
        )
    )

    long_reqs = getattr(cfg.retrieval, "long_requirements", None)
    long_launcher = find_launcher(long_reqs) if long_reqs else None

    if cfg.include_recall_100:
        cfg.evaluation.include_recall_100 = True

    # Built tests collections depending on config
    tests = build_tests(
        cfg.evaluation,
        retrievers_only=cfg.retrievers_only,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        long_evals=getattr(cfg.retrieval, "long_evals", None),
        check_docs=False,
        blocking=cfg.preprocessing.blocking_download,
    )

    # Built Retrievers - list of splade/plaid models or just bm25
    all_retrievers = []
    if len(cfg.retrievers_hf_id) > 0:
        for retriever_hf_id in cfg.retrievers_hf_id:
            if not retriever_hf_id:
                continue

            prepare_hf_model(retriever_hf_id)

            if cfg.use_plaid:
                logging.info(f"Instantiating PLAID retriever {retriever_hf_id} ")

                plaid_doc_max = (
                    cfg.plaid.max_doc_length
                    if cfg.plaid.max_doc_length is not None
                    else (cfg.max_doc_length or 300)
                )
                plaid_query_max = (
                    cfg.plaid.max_query_length
                    if cfg.plaid.max_query_length is not None
                    else (cfg.max_query_length or 32)
                )

                plaid_encoder = st_multivector_scorer(
                    model_id=retriever_hf_id,
                    doc_maxlen=plaid_doc_max,
                    query_maxlen=plaid_query_max,
                    pref_attn_implementation=cfg.attn_implementation,
                )
                plaid_encoder.tag("scorer", retriever_hf_id)
                plaid_encoder.tag("scorer_type", ScorerType.stMultiVectorEncoder.value)

                all_retrievers.append(
                    (
                        retriever_hf_id,
                        partial(
                            plaid_retriever,
                            cfg,
                            retriever_hf_id,
                            plaid_encoder,
                            launcher_index=launcher_index,
                            topk=cfg.retrieval.k,
                        ),
                        [],
                    )
                )
            else:
                logging.info(f"Instantiating SPLADE retriever {retriever_hf_id} ")

                splade_encoder, retriever_init_tasks = (
                    splade_encoder_from_pretrained_hf(retriever_hf_id)
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
                            topk=cfg.retrieval.k,
                            init_tasks=retriever_init_tasks,
                        ),
                        retriever_init_tasks,
                    )
                )

    if len(all_retrievers) == 0:
        # add bm25 by default
        all_retrievers.append(
            (
                "bm25",
                partial(bm25_retriever, cfg, "bm25", launcher_index=launcher_index),
                [],  # no init tasks for bm25
            )
        )

    logging.info(f"Evaluating retrievers: {[name for name, _, _ in all_retrievers]}")

    # Evaluate First stage retrievers only and store the results to reuse them with a second stage cross-encoder
    for retriever_name, retriever_factory, retriever_init_tasks in all_retrievers:
        # Eval First stage only
        eval_results = tests.evaluate_retriever(
            retriever_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
            fabric_config=fabric_config,
        )

        # Create a MultiRunRetrieverFactory storing results from first stage
        run_retriever_factory = MultiRunRetrieverFactory.from_results(
            retriever_name, eval_results
        )

        logging.info(
            f"First stage only evaluation done for {retriever_name} on datasets {list(run_retriever_factory.runs.keys())}"
        )
        logging.info(
            f"Evaluating model-based retrievers {cfg.scorers_hf_id} {cfg.scorers_st_id} {cfg.scorers_st_multivector_id}"
        )

        # Eval With cross-encoder
        if cfg.retrievers_only:
            if (
                len(cfg.scorers_hf_id) > 0
                or len(cfg.scorers_st_id) > 0
                or len(cfg.scorers_st_multivector_id) > 0
            ):
                logging.warning(
                    "Scorers specified in config but retrievers_only is True. Skipping cross-encoder evaluation."
                )
            continue

        all_scorers = (
            [(id, "hf") for id in cfg.scorers_hf_id]
            + [(id, "st") for id in cfg.scorers_st_id]
            + [
                (id, ScorerType.stMultiVectorEncoder.value)
                for id in cfg.scorers_st_multivector_id
            ]
        )

        for scorer_id, scorer_type in all_scorers:
            # Build the cross encoder
            prepare_hf_model(scorer_id)

            # Logic for max_length: use get_default_max_len for model id,
            # and pass it to constructor only if default is higher than the one given
            ce_init_tasks = []
            if scorer_type == "hf":
                scorer, ce_init_tasks = hf_cross_scorer(
                    hf_id=scorer_id,
                    max_length=cfg.max_length,
                    max_doc_length=cfg.max_doc_length,
                    max_query_length=cfg.max_query_length,
                    pref_attn_implementation=cfg.attn_implementation,
                )
            elif scorer_type == "st":
                st_tok = None
                if cfg.max_query_length is not None or cfg.max_doc_length is not None:
                    from xpmir.neural.huggingface import HFQueryDocTokenizer

                    st_tok = HFQueryDocTokenizer.C(
                        model_id=scorer_id,
                        max_query_length=cfg.max_query_length,
                        max_doc_length=cfg.max_doc_length,
                        max_length=cfg.max_length,
                    )

                scorer = st_cross_scorer(
                    model_id=scorer_id,
                    max_length=cfg.max_length,
                    pref_attn_implementation=cfg.attn_implementation,
                    tokenizer=st_tok,
                )
            elif scorer_type == ScorerType.stMultiVectorEncoder.value:
                query_maxlen = cfg.max_query_length
                doc_maxlen = cfg.max_doc_length or cfg.max_length

                if query_maxlen is not None and query_maxlen > 512:
                    logging.warning(
                        f"MultiVector scorer '{scorer_id}': max_query_length ({query_maxlen}) exceeds 512 tokens. "
                        "ColBERT queries are typically 32 tokens; consider checking your length config."
                    )
                if doc_maxlen is not None and doc_maxlen > 512:
                    logging.warning(
                        f"MultiVector scorer '{scorer_id}': max_doc_length ({doc_maxlen}) exceeds 512 tokens. "
                        "Consider checking your length config."
                    )

                scorer = st_multivector_scorer(
                    model_id=scorer_id,
                    query_maxlen=query_maxlen,
                    doc_maxlen=doc_maxlen,
                    pref_attn_implementation=cfg.attn_implementation,
                )
            else:
                raise ValueError(f"Unknown scorer_type: '{scorer_type}'")

            scorer.tag("scorer", scorer_id)
            scorer.tag("scorer_type", scorer_type)
            if cfg.attn_implementation:
                scorer.tag("attn_implementation", cfg.attn_implementation)

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
                fabric_config=fabric_config,
            )

    # Wait for all tasks to complete
    helper.xp.wait()
    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    # Identify available metric columns and convert to numeric
    all_metric_cols = [col for col in df.columns if col[0] == "metric"]
    df[all_metric_cols] = df[all_metric_cols].apply(pd.to_numeric, downcast="float")

    # Flatten MultiIndex columns and remove duplicates (e.g., 'dataset' might be both a tag and a column)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    cols_to_drop = [col for col in df.columns if "index_doc" in str(col).lower()]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if cfg.retrievers_only:
        tag_names = ["first_stage"]
    else:
        tag_names = ["first_stage", "scorer", "scorer_type"]
    group_by_tags = [col for col in sorted(list(tag_names)) if col in df.columns]

    # Add aggregations
    df = add_dataset_aggregations(
        df,
        group_by_cols=group_by_tags,
        aggregations=aggregations,
        add_mean=True,
    )

    # Reorder columns: tags first, metrics after
    tag_cols = ["dataset"] + group_by_tags
    tag_cols = [c for c in tag_cols if c in df.columns]
    metric_cols = [c for c in df.columns if c not in tag_cols]
    df = df[tag_cols + metric_cols]

    logging.info(f"Final DataFrame:\n{df}")

    # save results
    if helper.xp.run_mode == RunMode.NORMAL:
        if not helper.xp.resultspath.exists():
            helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

        output_file = helper.xp.resultspath / "results.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

        # Generate and save LaTeX table
        metric_col = "nDCG@10" if not cfg.retrievers_only else "R@1000"
        if metric_col not in df.columns:
            for fallback in ["nDCG@10", "R@100", "R@1000", "RR@10", "Success@5"]:
                if fallback in df.columns:
                    metric_col = fallback
                    break

        latex_table = dataframe_to_latex(
            df,
            caption="Evaluation Results",
            label="tab:eval_results",
            sig_df=None,
            metric_col=metric_col,
        )
        latex_output_file = helper.xp.resultspath / "results.tex"
        with open(latex_output_file, "w") as f:
            f.write(latex_table)
        logging.info(f"LaTeX table saved to {latex_output_file}")
    else:
        logging.info("Dry-run mode: skipping saving results to disk")
