"""Utilities for index management."""

import logging
from pathlib import Path
import subprocess
from typing import Optional, List, Dict, Tuple
from experimaestro import Launcher, Task
from xpmir.index.sparse import (
    SparseRetrieverIndexBuilder,
    AbstractSparseRetrieverIndexBuilder,
)
from xpmir.index.plaid import PlaidIndexBuilder
from xpmir.rankers import Documents
from xpm_torch.configuration import FabricConfiguration
from configuration import Indexation, PlaidConfiguration

logger = logging.getLogger(__name__)

# cache the indexes
_indexes = {}


def get_splade_index(
    documents: Documents,
    splade_encoder,
    indexation_cfg,
    launcher_index,
    init_tasks: list = [],
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
):
    """Build an index for given documents, using a given Sparse retriever model
    Caches it to avoid submitting job twice.
    """
    triggers = (
        long_evals
        if long_evals is not None
        else (
            getattr(indexation_cfg, "long_evals", None)
            if indexation_cfg is not None
            else None
        )
    ) or ["fever", "msmarco", "hotpotqa"]

    target_key = key or documents.id
    is_long = bool(
        long_launcher_index is not None
        and any(trigger in target_key for trigger in triggers)
    )
    launcher = long_launcher_index if is_long else launcher_index

    index_cfg = SparseRetrieverIndexBuilder.C(
        batch_size=indexation_cfg.batch_size,
        # batcher=PowerAdaptativeBatcher.C(),
        encoder=splade_encoder,
        documents=documents,
        ordered_index=False,
        max_docs=indexation_cfg.max_indexed,
    )

    indexer_id = index_cfg.__identifier__()

    if indexer_id not in _indexes:
        logger.info(
            "Indexing %s (key=%s, %s documents) with %s: %s",
            documents.id,
            key,
            documents.count,
            "long_launcher_index" if is_long else "launcher_index",
            launcher,
        )
        index = index_cfg.tag("index_documents", documents.id).submit(
            launcher=launcher, init_tasks=init_tasks
        )
        _indexes[indexer_id] = index
    else:
        index = _indexes[indexer_id]

    return index


def get_plaid_index(
    documents: Documents,
    encoder,
    plaid_cfg: PlaidConfiguration,
    launcher_index,
    indexation_cfg: Optional[Indexation] = None,
    batch_size: Optional[int] = None,
    init_tasks: list = None,
    seed: int = 42,
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
):
    """Build a PLAID index for given documents using a given multi-vector/late-interaction encoder.
    Caches it to avoid submitting the job twice.
    """
    if init_tasks is None:
        init_tasks = []

    triggers = (
        long_evals
        if long_evals is not None
        else (
            getattr(indexation_cfg, "long_evals", None)
            if indexation_cfg is not None
            else None
        )
    ) or ["fever", "msmarco", "hotpotqa"]

    target_key = key or documents.id
    is_long = bool(
        long_launcher_index is not None
        and any(trigger in target_key for trigger in triggers)
    )
    launcher = long_launcher_index if is_long else launcher_index

    # Document encoder batch size (GPU forward pass)
    if batch_size is not None and batch_size > 0:
        doc_batch_size = batch_size
    elif indexation_cfg is not None and indexation_cfg.batch_size > 0:
        doc_batch_size = indexation_cfg.batch_size
    else:
        doc_batch_size = 32

    # Fast-PLAID internal token batch size
    fast_plaid_bs = (
        plaid_cfg.fast_plaid_batch_size
        if plaid_cfg.fast_plaid_batch_size > 0
        else 25000
    )

    index_builder = PlaidIndexBuilder.C(
        documents=documents,
        encoder=encoder,
        buffer_size=plaid_cfg.buffer_size,
        batch_size=doc_batch_size,
        fast_plaid_batch_size=fast_plaid_bs,
        n_bits=plaid_cfg.n_bits,
        kmeans_niters=plaid_cfg.kmeans_niters,
        n_samples_kmeans=plaid_cfg.n_samples_kmeans,
        max_points_per_centroid=plaid_cfg.max_points_per_centroid,
        seed=seed,
        compress_only=plaid_cfg.compress_only,
        force_cpu_indexing=plaid_cfg.force_cpu_indexing,
        fabric_config=FabricConfiguration.C(precision="bf16-true"),
    )

    init_paths = tuple(str(getattr(t, "path", t)) for t in init_tasks)
    cache_key = (index_builder.__identifier__(), init_paths)

    if cache_key not in _indexes:
        logger.info(
            "Indexing %s (key=%s, %s documents) with PLAID via %s: %s",
            documents.id,
            key,
            getattr(documents, "count", "unknown"),
            "long_launcher_index" if is_long else "launcher_index",
            launcher,
        )
        index = index_builder.tag("index_documents", documents.id).submit(
            launcher=launcher, init_tasks=init_tasks
        )
        _indexes[cache_key] = index
    else:
        index = _indexes[cache_key]

    return index


def get_path_size_mb(path: Path) -> float:
    """Queries on-disk size of a directory or file in Megabytes using `du -sk` with Python fallback."""
    if not path.exists():
        raise FileNotFoundError(f"Index path does not exist: {path}")

    try:
        proc = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        kb = int(proc.stdout.split()[0])
        return kb / 1024.0
    except Exception as e:
        logger.warning(
            f"`du -sk` failed on {path} ({e}); falling back to recursive file scan"
        )
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total_bytes / (1024.0 * 1024.0)


def find_index_builders(
    evaluation, builder_classes: Optional[Tuple[type, ...]] = None
) -> List[Task]:
    """Traverses the experimaestro dependency graph from an evaluation task or output
    to find all upstream PlaidIndexBuilder and SparseRetrieverIndexBuilder instances."""
    if builder_classes is None:
        builder_classes = (PlaidIndexBuilder, AbstractSparseRetrieverIndexBuilder)

    task = (
        getattr(getattr(evaluation, "__xpm__", None), "task", None)
        or getattr(evaluation, "task", None)
        or evaluation
    )
    job = getattr(getattr(task, "__xpm__", None), "job", None) or getattr(
        task, "job", None
    )
    if job is None:
        logger.warning(
            f"Could not locate experimaestro job for evaluation {evaluation}"
        )
        return []

    visited_jobs = set()
    builders: List[Task] = []
    seen_builder_ids = set()

    def dfs(current_job):
        if id(current_job) in visited_jobs:
            return
        visited_jobs.add(id(current_job))

        cfg = getattr(current_job, "config", None)
        if cfg is not None and isinstance(cfg, builder_classes):
            if id(cfg) not in seen_builder_ids:
                seen_builder_ids.add(id(cfg))
                builders.append(cfg)

        for dep in getattr(current_job, "dependencies", []):
            origin_job = getattr(dep, "origin", None)
            if origin_job is not None:
                dfs(origin_job)

    dfs(job)
    return builders


def sync_index_size_metrics(
    tests,
    builder_classes: Optional[Tuple[type, ...]] = None,
) -> None:
    """Standalone function to query on-disk index sizes for all Plaid and Sparse
    retriever index builders in the evaluate task dependencies, syncing <ClassName>_index_size
    into each evaluation's aggregated.txt file before to_dataframe()."""
    if not hasattr(tests, "collection"):
        raise AttributeError(
            f"Provided tests object ({type(tests)}) has no 'collection' attribute"
        )

    size_cache: Dict[str, float] = {}

    for dataset_name, evaluations in tests.collection.items():
        eval_list = getattr(evaluations, "results", [])
        if not eval_list and hasattr(evaluations, "per_tags"):
            eval_list = list(evaluations.per_tags.values())

        for evaluation in eval_list:
            results_path = getattr(evaluation, "results", None)
            if results_path is None:
                continue
            results_path = Path(results_path)
            if not results_path.exists():
                logger.warning(
                    f"Aggregated results file {results_path} not found for dataset {dataset_name}"
                )
                continue

            builders = find_index_builders(evaluation, builder_classes=builder_classes)
            if not builders:
                logger.debug(
                    f"No index builders found in dependencies for evaluation on dataset {dataset_name}"
                )
                continue

            existing_text = results_path.read_text()
            existing_lines = existing_text.splitlines()
            metrics_to_write = {}

            for builder in builders:
                orig_type = getattr(
                    getattr(builder, "__xpmtype__", None),
                    "originaltype",
                    builder.__class__,
                )
                class_name = orig_type.__name__
                metric_name = f"index_size_{class_name}"

                index_path = getattr(builder, "index_path", None)
                if index_path is None:
                    logger.warning(
                        f"Builder {builder} ({class_name}) has no 'index_path' attribute"
                    )
                    continue

                index_path = Path(index_path)
                cache_key = str(index_path)
                if cache_key not in size_cache:
                    size_cache[cache_key] = get_path_size_mb(index_path)
                    logger.info(
                        f"Measured index size for {class_name} ({index_path}): {size_cache[cache_key]:.2f} MB"
                    )

                size_mb = size_cache[cache_key]
                metrics_to_write[metric_name] = size_mb

            # Update existing lines in-place, or append new metric lines
            updated_lines = []
            seen_metrics = set()
            for line in existing_lines:
                tokens = line.strip().split()
                if tokens and tokens[0] in metrics_to_write:
                    metric = tokens[0]
                    size_mb = metrics_to_write[metric]
                    updated_lines.append(f"{metric:40s} {'all':10s} {size_mb:.4f}")
                    seen_metrics.add(metric)
                elif line.strip():
                    updated_lines.append(line)

            for metric, size_mb in metrics_to_write.items():
                if metric not in seen_metrics:
                    updated_lines.append(f"{metric:40s} {'all':10s} {size_mb:.4f}")

            new_text = "\n".join(updated_lines) + "\n"
            if new_text != existing_text:
                results_path.write_text(new_text)
                logger.info(
                    f"Updated {len(metrics_to_write)} index size metric(s) in {results_path}"
                )
