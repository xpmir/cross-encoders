"""
MICE V2 (ModernBERT-based) Training Experiment.
Implements training pipeline for MICE V2 models.
"""

import logging
from typing import List, Tuple, Optional

from experimaestro import LightweightTask
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment

from xpmir.papers.results import PaperResults
from xpmir.rankers import AbstractModuleScorer
from xpmir.text.huggingface.tokenizers import get_default_max_len
from xpmir.papers import configuration

from MICE.experiments.mice_training import TrainEvaluateLateInteractionScorer
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer
from configuration import CE_FineTuning, GridSearch
from format import loss_names, backbone_names_lower
from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class MiceV2_FineTuning(CE_FineTuning):
    ## MICE V2 specific configuration
    n_contextualization_layers: GridSearch[int] = 4
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: GridSearch[Optional[int]] = None
    """Number of bottom encoder layers for the document. If None, use n_contextualization_layers."""

    n_interaction_layers: GridSearch[Optional[int]] = None
    """Number of top encoder layers with cross-attention. If None, use all remaining layers from backbone."""

    bound_bottom_layers: GridSearch[bool] = True
    """Whether to bound bottom query and document encoding layers"""


def build_mice_v2(
    cfg: MiceV2_FineTuning,
) -> Tuple[AbstractModuleScorer, List[LightweightTask]]:
    """Build the MICE V2 model using the modernbert_xpm factory."""
    default_max_len = get_default_max_len(cfg.base)
    if cfg.max_length and default_max_len > cfg.max_length:
        max_len = cfg.max_length
    else:
        max_len = default_max_len

    return mice_v2_scorer(
        hf_id=cfg.base,
        n_contextualization_layers=cfg.n_contextualization_layers,
        n_docs_ctx_layers=cfg.n_docs_ctx_layers,
        n_interaction_layers=cfg.n_interaction_layers,
        bound_bottom_layers=cfg.bound_bottom_layers,
        max_length=max_len,
    )


def get_name_from_tags(model_tags: dict, cfg: MiceV2_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    n_ctx_layers = model_tags.get(
        "n_contextualization_layers", cfg.n_contextualization_layers
    )
    n_docs_layers = model_tags.get("n_docs_ctx_layers", cfg.n_docs_ctx_layers)
    doc_suffix = f"-d{n_docs_layers}" if n_docs_layers is not None else ""

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"MiceV2-l{n_ctx_layers}{doc_suffix}-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: MiceV2_FineTuning) -> PaperResults:
    """Main entry point for MICE V2 training experiment."""
    return TrainEvaluateLateInteractionScorer(
        helper=helper,
        cfg=cfg,
        build_model_fn=build_mice_v2,
        get_model_name_fn=get_name_from_tags,
    )
