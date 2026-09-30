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

from typing import List, Optional, Dict, Any, Tuple
from attrs import Factory, field
from functools import partial
import pandas as pd
from pathlib import Path

from experimaestro import RunMode, stop_tags
from experimaestro.launcherfinder import find_launcher
from experimaestro.experiments.grid import generate_grid, GridSearch
from experimaestro.core.objects.config import clone

from xpm_torch.huggingface import prepare_hf_model
from xpm_torch.configuration import FabricConfiguration

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration
from xpmir.experiments.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.neural.sentence_transformers import (
    st_cross_scorer,
    st_multivector_scorer,
    STCrossEncoder,
    STMultiVectorEncoder,
)
from xpmir.rankers import scorer_retriever
from xpmir.rankers.scorer import Scorer
from xpmir.evaluation import MultiRunRetrieverFactory

from format import dataframe_to_latex, aggregations
from tests import build_tests, EvaluationsCollection
from configuration import (
    Retrieval,
    Indexation,
    Preprocessing,
    Evaluation,
    ScorerType,
    PlaidConfiguration,
)
from retrievers import (
    splade_retriever,
    bm25_retriever,
    plaid_retriever,
    MicePlaidRetrieverFactory,
)
from training_utils import (
    add_dataset_aggregations,
    resolve_trained_model_loader,
    parse_experiment_ref,
    sync_eval_sparsity_metrics,
    find_candidate_experiment_dirs,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    ModernBertMiceScorer,
    InitEvalSparsityHook,
)

import logging

from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)
    preprocessing: Preprocessing = Factory(Preprocessing)

    scorer_id: GridSearch[str] = ""
    """Identifier for the scorer model: Hugging Face model ID, ST model ID, local folder, or experiment ID."""

    scorer_type: GridSearch[str] = ScorerType.stCrossEncoder.value
    """Type of scorer model: stCrossEncoder, hfCrossScorer, stMultiVectorEncoder, miceV2, or from_experiment."""

    from_experiment: GridSearch[Optional[str]] = None
    """Optional experiment ID (e.g. 'exp_id' or 'exp_id/run_id' to pin a specific run), local model directory path, or HF Hub model ID to load a trained model."""

    from_run_id: Optional[str] = None
    """Optional run ID within from_experiment (e.g. '20260924_180227')."""

    attn_implementation: GridSearch[Optional[str]] = None
    """Preferred attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')."""

    precision: Optional[str] = None
    """Precision setting for evaluation (e.g. 'bf16-mixed')."""

    max_length: GridSearch[Optional[int]] = None
    """Maximum length for cross-encoders."""

    max_query_length: GridSearch[Optional[int]] = None
    """Maximum query length for cross-encoders."""

    max_doc_length: GridSearch[Optional[int]] = None
    """Maximum document length for cross-encoders."""

    query_prompt: GridSearch[Optional[str]] = None
    """Optional query prompt template or prefix (e.g. '[Q] ' or 'query: ')."""

    doc_prompt: GridSearch[Optional[str]] = None
    """Optional document prompt template or prefix (e.g. '[D] ' or 'passage: ')."""

    hard_pruning: GridSearch[Optional[bool]] = None
    """Whether to enforce hard token pruning for MICE models during evaluation."""

    hard_prune_threshold: GridSearch[Optional[float]] = None
    """Utility threshold below which tokens are hard-pruned for MICE models."""

    eval_sparsity: bool = False
    """Whether to compute token sparsity and retention ratio during evaluation benchmarks."""

    save_runs: bool = False
    """Whether to save the evaluation runs in the resultspath."""

    retrievers_hf_id: List[str] = field(factory=list)
    """List of first-stage retrievers to evaluate (SPLADE or PLAID). If empty, uses BM25."""

    use_plaid: bool = False
    """If true, evaluate retrievers_hf_id using PLAID multi-vector retrieval instead of SPLADE."""

    plaid: PlaidConfiguration = Factory(PlaidConfiguration)
    """Configuration for PLAID indexing and retrieval."""

    include_recall_100: bool = False
    """Whether to additionally report R@100 alongside CE_MEASURES."""

    evaluation: Evaluation = Factory(Evaluation)

    grid_search: Dict[str, Any] = field(factory=dict)
    """Grid search parameters mapping parameter paths to values or config_dicts combinations."""


def build_first_stage_retrievers(
    cfg: BaselinesConfig,
    launcher_index,
    cfg_tags: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Any, List[Any]]]:
    """Instantiate first-stage retrievers (SPLADE, PLAID, or BM25)."""
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
                            tags=cfg_tags,
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
    return all_retrievers


def resolve_evaluation_scorer(
    exp_ref: str,
    cfg: BaselinesConfig,
) -> Tuple[Scorer, List[Any], Any, str]:
    """Resolves an experiment ID, local path, or HF Hub model, clones it for safe
    parameter overrides, and returns the configured Scorer and its init tasks."""
    loader = resolve_trained_model_loader(exp_ref)
    source_model = loader.model
    scorer = clone(source_model)

    is_mice_v2 = isinstance(scorer, ModernBertMiceScorer)
    is_st_mv = isinstance(scorer, STMultiVectorEncoder)
    is_st_ce = isinstance(scorer, STCrossEncoder)

    # Length overrides
    if cfg.max_length is not None:
        if hasattr(scorer, "max_length"):
            scorer.max_length = cfg.max_length
        if hasattr(scorer, "tokenizer") and hasattr(scorer.tokenizer, "max_length"):
            scorer.tokenizer.max_length = cfg.max_length
    if cfg.max_query_length is not None:
        if hasattr(scorer, "max_query_length"):
            scorer.max_query_length = cfg.max_query_length
        if hasattr(scorer, "tokenizer") and hasattr(
            scorer.tokenizer, "max_query_length"
        ):
            scorer.tokenizer.max_query_length = cfg.max_query_length
    if cfg.max_doc_length is not None:
        if hasattr(scorer, "max_doc_length"):
            scorer.max_doc_length = cfg.max_doc_length
        if hasattr(scorer, "tokenizer") and hasattr(scorer.tokenizer, "max_doc_length"):
            scorer.tokenizer.max_doc_length = cfg.max_doc_length

    # Prompt overrides
    if (
        cfg.query_prompt is not None
        and hasattr(scorer, "tokenizer")
        and hasattr(scorer.tokenizer, "query_prompt")
    ):
        scorer.tokenizer.query_prompt = cfg.query_prompt
    if (
        cfg.doc_prompt is not None
        and hasattr(scorer, "tokenizer")
        and hasattr(scorer.tokenizer, "doc_prompt")
    ):
        scorer.tokenizer.doc_prompt = cfg.doc_prompt

    # Attention implementation
    if cfg.attn_implementation is not None:
        if hasattr(scorer, "attn_implementation"):
            scorer.attn_implementation = cfg.attn_implementation
        elif hasattr(scorer, "pref_attn_implementation"):
            scorer.pref_attn_implementation = cfg.attn_implementation

    # MICE v2 pruning overrides
    if is_mice_v2:
        if cfg.hard_pruning is not None:
            scorer.hard_pruning = cfg.hard_pruning
        elif cfg.hard_prune_threshold is not None:
            scorer.hard_pruning = True
        if cfg.hard_prune_threshold is not None:
            scorer.hard_prune_threshold = cfg.hard_prune_threshold

    # Build independent loader
    model_loader = scorer.loader_config(
        loader.path, settings=getattr(loader, "settings", None)
    )
    init_tasks = [stop_tags(model_loader)]

    if cfg.eval_sparsity and is_mice_v2:
        init_tasks.append(
            InitEvalSparsityHook.C(
                scorer=scorer,
                threshold=getattr(scorer, "hard_prune_threshold", 0.5),
            )
        )

    resolved_type_str = (
        "mice_v2"
        if is_mice_v2
        else ("st_multivector" if is_st_mv else ("st" if is_st_ce else "hf"))
    )
    return scorer, init_tasks, model_loader, resolved_type_str


def evaluate_scorer(
    cfg: BaselinesConfig,
    cfg_tags: Dict[str, Any],
    tests: EvaluationsCollection,
    first_stage_factories: Dict[str, Tuple[MultiRunRetrieverFactory, List[Any]]],
    launcher_evaluate,
    launcher_index,
    fabric_config,
):
    """Evaluates a single scorer configuration over all precomputed first-stage retrievers."""
    target_id = ""
    is_from_exp = False
    if cfg.from_experiment:
        target_id = cfg.from_experiment
        if cfg.from_run_id and "/" not in target_id:
            target_id = f"{target_id}/{cfg.from_run_id}"
        is_from_exp = True
    elif cfg.scorer_id:
        target_id = cfg.scorer_id
        exp_id, _ = parse_experiment_ref(target_id)
        if Path(target_id).is_dir() or bool(find_candidate_experiment_dirs(exp_id)):
            is_from_exp = True
    else:
        raise ValueError(
            "Neither 'scorer_id' nor 'from_experiment' was specified in config."
        )

    scorer_type_enum = ScorerType.from_string(cfg.scorer_type)
    if scorer_type_enum == ScorerType.from_experiment:
        is_from_exp = True

    model_loader = None
    if is_from_exp or scorer_type_enum in (
        ScorerType.miceV2,
        ScorerType.from_experiment,
    ):
        scorer, ce_init_tasks, model_loader, resolved_type_str = (
            resolve_evaluation_scorer(target_id, cfg)
        )
    elif scorer_type_enum == ScorerType.hfCrossScorer:
        prepare_hf_model(target_id)
        scorer, ce_init_tasks = hf_cross_scorer(
            hf_id=target_id,
            max_length=cfg.max_length,
            max_doc_length=cfg.max_doc_length,
            max_query_length=cfg.max_query_length,
            pref_attn_implementation=cfg.attn_implementation,
        )
        resolved_type_str = ScorerType.hfCrossScorer.value
    elif scorer_type_enum == ScorerType.stCrossEncoder:
        prepare_hf_model(target_id)
        st_tok = None
        if cfg.max_query_length is not None or cfg.max_doc_length is not None:
            from xpmir.neural.huggingface import HFQueryDocTokenizer

            st_tok = HFQueryDocTokenizer.C(
                model_id=target_id,
                max_query_length=cfg.max_query_length,
                max_doc_length=cfg.max_doc_length,
                max_length=cfg.max_length,
            )
        scorer = st_cross_scorer(
            model_id=target_id,
            max_length=cfg.max_length,
            pref_attn_implementation=cfg.attn_implementation,
            tokenizer=st_tok,
        )
        ce_init_tasks = []
        resolved_type_str = ScorerType.stCrossEncoder.value
    elif scorer_type_enum == ScorerType.stMultiVectorEncoder:
        prepare_hf_model(target_id)
        query_maxlen = cfg.max_query_length
        doc_maxlen = cfg.max_doc_length or cfg.max_length

        if query_maxlen is not None and query_maxlen > 512:
            logging.warning(
                f"MultiVector scorer '{target_id}': max_query_length ({query_maxlen}) exceeds 512 tokens. "
                "ColBERT queries are typically 32 tokens; consider checking your length config."
            )
        if doc_maxlen is not None and doc_maxlen > 512:
            logging.warning(
                f"MultiVector scorer '{target_id}': max_doc_length ({doc_maxlen}) exceeds 512 tokens. "
                "Consider checking your length config."
            )

        scorer = st_multivector_scorer(
            model_id=target_id,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            pref_attn_implementation=cfg.attn_implementation,
        )
        ce_init_tasks = []
        resolved_type_str = ScorerType.stMultiVectorEncoder.value
    else:
        raise ValueError(f"Unsupported scorer_type: '{scorer_type_enum}'")

    scorer.tag("scorer", target_id)
    scorer.tag("scorer_type", resolved_type_str)
    if is_from_exp:
        exp_id, run_id = parse_experiment_ref(target_id)
        scorer.tag("from_experiment", exp_id)
        if run_id:
            scorer.tag("from_run_id", run_id)
    if getattr(scorer, "hard_prune_threshold", None) is not None:
        scorer.tag("prune_threshold", str(scorer.hard_prune_threshold))
    if cfg.attn_implementation:
        scorer.tag("attn_implementation", cfg.attn_implementation)

    for k, v in cfg_tags.items():
        if k not in ("scorer", "scorer_type", "from_experiment"):
            scorer.tag(k, v)

    for retriever_name, (run_retriever_factory, _) in first_stage_factories.items():
        is_mice_plaid = cfg.plaid.use_plaid and resolved_type_str == "mice_v2"

        if not is_mice_plaid or cfg.compare_with_baseline:
            logging.info(
                f"Scheduling 2-stage evaluation: first_stage={retriever_name}, "
                f"scorer={target_id} ({resolved_type_str})"
            )
            two_stage_retriever_factory = partial(
                scorer_retriever,
                batch_size=cfg.retrieval.batch_size,
                scorer=scorer,
                retrievers=run_retriever_factory,
            )
            tests.evaluate_retriever(
                two_stage_retriever_factory,
                launcher=launcher_evaluate,
                init_tasks=ce_init_tasks,
                fabric_config=fabric_config,
                with_run=cfg.save_runs,
            )

        if is_mice_plaid:
            logging.info(f"Scheduling MICE-PLAID evaluation for: {target_id}")
            plaid_scorer = clone(scorer)
            plaid_scorer.tag("scorer", target_id)
            plaid_scorer.tag("scorer_type", resolved_type_str)
            plaid_scorer.tag("plaid_retriever", True)
            if is_from_exp:
                exp_id, run_id = parse_experiment_ref(target_id)
                plaid_scorer.tag("from_experiment", exp_id)
                if run_id:
                    plaid_scorer.tag("from_run_id", run_id)
            if getattr(plaid_scorer, "hard_prune_threshold", None) is not None:
                plaid_scorer.tag(
                    "prune_threshold", str(plaid_scorer.hard_prune_threshold)
                )
            for k, v in cfg_tags.items():
                if k not in (
                    "scorer",
                    "scorer_type",
                    "from_experiment",
                    "plaid_retriever",
                ):
                    plaid_scorer.tag(k, v)

            plaid_loader = plaid_scorer.loader_config(
                model_loader.path,
                settings=getattr(model_loader, "settings", None),
            )
            plaid_init_tasks = [stop_tags(plaid_loader)]

            mice_plaid_factory = MicePlaidRetrieverFactory(
                mice_model=plaid_scorer,
                first_stage_factory=run_retriever_factory,
                topk=cfg.retrieval.k,
                batchsize=cfg.retrieval.batch_size,
                n_ivf_probe=cfg.plaid.n_ivf_probe,
                n_full_scores=cfg.plaid.n_full_scores,
                max_query_length=cfg.plaid.max_query_length,
                max_doc_length=cfg.plaid.max_doc_length,
            )
            mice_plaid_factory.build_indexes(
                evaluations=tests,
                cfg=cfg.plaid,
                batch_size=cfg.retrieval.batch_size,
                seed=0,
                launcher_index=launcher_index,
                init_tasks=plaid_init_tasks,
            )
            tests.evaluate_retriever(
                retriever=mice_plaid_factory,
                launcher=launcher_evaluate,
                init_tasks=plaid_init_tasks,
                fabric_config=fabric_config,
                with_run=cfg.save_runs,
            )

            if cfg.plaid.use_colbert_first_stage:
                logging.info(
                    f"Scheduling MICE-ColBERT first-stage evaluation for: {target_id}"
                )
                colbert_runs = tests.evaluate_retriever(
                    mice_plaid_factory.colbert_retriever,
                    launcher=launcher_evaluate,
                    init_tasks=plaid_init_tasks,
                    with_run=True,
                    fabric_config=fabric_config,
                )
                colbert_run_factory = MultiRunRetrieverFactory.from_results(
                    "mice-colbert", colbert_runs
                )
                mice_colbert_rerank_factory = MicePlaidRetrieverFactory(
                    mice_model=plaid_scorer,
                    first_stage_factory=colbert_run_factory,
                    topk=cfg.retrieval.k,
                    batchsize=cfg.retrieval.batch_size,
                    n_ivf_probe=cfg.plaid.n_ivf_probe,
                    n_full_scores=cfg.plaid.n_full_scores,
                    max_query_length=cfg.plaid.max_query_length,
                    max_doc_length=cfg.plaid.max_doc_length,
                )
                mice_colbert_rerank_factory.build_indexes(
                    evaluations=tests,
                    cfg=cfg.plaid,
                    batch_size=cfg.retrieval.batch_size,
                    seed=0,
                    launcher_index=launcher_index,
                    init_tasks=plaid_init_tasks,
                )
                tests.evaluate_retriever(
                    retriever=mice_colbert_rerank_factory,
                    launcher=launcher_evaluate,
                    init_tasks=plaid_init_tasks,
                    fabric_config=fabric_config,
                    with_run=cfg.save_runs,
                )


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: BaselinesConfig) -> PaperResults:
    all_configs, all_tags = generate_grid(cfg)
    if not all_configs:
        raise ValueError(
            f"generate_grid returned 0 configurations for {getattr(cfg, 'id', 'evaluation')}. "
            "Please check your grid_search configuration or parameter definitions."
        )

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

    base_config = all_configs[0]
    launcher_evaluate = find_launcher(base_config.retrieval.requirements)
    launcher_index = find_launcher(base_config.indexation.requirements)
    launcher_preprocessing = find_launcher(base_config.preprocessing.requirements)

    fabric_config = (
        base_config.evaluation.fabric.get_config()
        if hasattr(base_config.evaluation, "fabric")
        else (
            FabricConfiguration.C(precision=base_config.precision)
            if getattr(base_config, "precision", None)
            else None
        )
    )

    long_reqs = getattr(base_config.retrieval, "long_requirements", None)
    long_launcher = find_launcher(long_reqs) if long_reqs else None

    if base_config.include_recall_100:
        base_config.evaluation.include_recall_100 = True

    # Build tests collections ONCE
    tests = build_tests(
        base_config.evaluation,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        long_evals=getattr(base_config.retrieval, "long_evals", None),
        check_docs=False,
        blocking=base_config.preprocessing.blocking_download,
    )

    has_scorers = any(bool(c.scorer_id or c.from_experiment) for c in all_configs)

    if not has_scorers:
        logging.info(
            "No scorers configured; evaluating retrievers directly across grid search."
        )
        for cfg_item, cfg_tags in zip(all_configs, all_tags):
            retrievers = build_first_stage_retrievers(
                cfg_item, launcher_index, cfg_tags=cfg_tags
            )
            for retriever_name, retriever_factory, retriever_init_tasks in retrievers:
                tests.evaluate_retriever(
                    retriever_factory,
                    launcher=launcher_evaluate,
                    init_tasks=retriever_init_tasks,
                    with_run=cfg_item.save_runs,
                    fabric_config=fabric_config,
                )
    else:
        # Build first-stage retrievers (SPLADE, PLAID, or BM25)
        all_retrievers = build_first_stage_retrievers(base_config, launcher_index)
        logging.info(
            f"Evaluating first-stage retrievers: {[name for name, _, _ in all_retrievers]}"
        )

        # Evaluate First stage retrievers only and cache their runs
        first_stage_factories: Dict[
            str, Tuple[MultiRunRetrieverFactory, List[Any]]
        ] = {}
        for retriever_name, retriever_factory, retriever_init_tasks in all_retrievers:
            eval_results = tests.evaluate_retriever(
                retriever_factory,
                launcher=launcher_evaluate,
                init_tasks=retriever_init_tasks,
                with_run=True,
                fabric_config=fabric_config,
            )
            run_retriever_factory = MultiRunRetrieverFactory.from_results(
                retriever_name, eval_results
            )
            first_stage_factories[retriever_name] = (
                run_retriever_factory,
                retriever_init_tasks,
            )
            logging.info(
                f"First stage only evaluation done for {retriever_name} on datasets {list(run_retriever_factory.runs.keys())}"
            )

        # Evaluate scorers across all grid configurations
        for cfg_item, cfg_tags in zip(all_configs, all_tags):
            evaluate_scorer(
                cfg_item,
                cfg_tags=cfg_tags,
                tests=tests,
                first_stage_factories=first_stage_factories,
                launcher_evaluate=launcher_evaluate,
                launcher_index=launcher_index,
                fabric_config=fabric_config,
            )

    # Wait for all tasks to complete
    helper.xp.wait()

    if any(getattr(c, "eval_sparsity", False) for c in all_configs):
        sync_eval_sparsity_metrics(tests)

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

    if not has_scorers:
        tag_names = ["first_stage"]
    else:
        tag_names = [
            "first_stage",
            "scorer",
            "scorer_type",
            "from_experiment",
            "from_run_id",
            "prune_threshold",
            "plaid_retriever",
        ]
    for t_dict in all_tags:
        for k in t_dict.keys():
            if k not in tag_names:
                tag_names.append(k)
    group_by_tags = [col for col in tag_names if col in df.columns]

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
        metric_col = "nDCG@10" if has_scorers else "R@100"
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
