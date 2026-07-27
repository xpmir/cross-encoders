"""
Experimaestro-IR integration for ModernBERT MICE V2 architecture.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import logging

import torch
import torch.nn as nn

from experimaestro import LightweightTask, Param, Constant, field
from xpmir.rankers.scorer import AbstractModuleScorer
from xpmir.letor.records import BaseItems
from xpm_torch.utils import to_device
from xpm_torch.module import SimpleModuleLoader

from MICE.modeling.mice import MICEQueryDocTokenizer, MICETokenizedTexts, QueryDocInput
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from transformers import AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class InitMICEV2ModernBERTFromHFID(LightweightTask):
    """Task to initialize weights of ModernBertMiceForSequenceClassification from HF checkpoint"""

    scorer: Param["ModernBertMiceScorer"]

    def execute(self):
        scorer = self.scorer
        logger.info(f"Loading MICE ModernBERT weights from {scorer.hf_id}")
        scorer.initialize()
        full_backbone = AutoModelForSequenceClassification.from_pretrained(scorer.hf_id)

        # 1. Embeddings
        if hasattr(full_backbone.model, "embeddings"):
            logger.info("Seeding embeddings from backbone")
            scorer.hf_model.model.embeddings.load_state_dict(
                full_backbone.model.embeddings.state_dict()
            )

        # 2. Bottom contextualization layers
        if hasattr(full_backbone.model, "layers"):
            logger.info("Seeding bottom layers from backbone")
            for i in range(scorer.n_contextualization_layers):
                if i < len(full_backbone.model.layers):
                    scorer.hf_model.model.layers[i].load_state_dict(
                        full_backbone.model.layers[i].state_dict()
                    )

        # 3. Top interaction layers
        if hasattr(full_backbone.model, "layers"):
            logger.info(
                f"Seeding {len(scorer.hf_model.top_layers)} top layers from backbone"
            )
            for idx, top_layer in enumerate(scorer.hf_model.top_layers):
                src_layer_idx = scorer.n_contextualization_layers + idx
                if src_layer_idx < len(full_backbone.model.layers):
                    src_layer = full_backbone.model.layers[src_layer_idx]
                    top_layer.self_attn.load_state_dict(src_layer.attn.state_dict())
                    top_layer.cross_attn.Wqkv.load_state_dict(
                        src_layer.attn.Wqkv.state_dict()
                    )
                    top_layer.cross_attn.Wo.load_state_dict(
                        src_layer.attn.Wo.state_dict()
                    )
                    top_layer.mlp.load_state_dict(src_layer.mlp.state_dict())

        # 4. Rotary embeddings
        if hasattr(full_backbone.model, "rotary_emb"):
            logger.info("Seeding rotary_emb from backbone")
            scorer.hf_model.model.rotary_emb.load_state_dict(
                full_backbone.model.rotary_emb.state_dict()
            )

        # 5. Final LayerNorm
        if hasattr(full_backbone.model, "final_norm"):
            logger.info("Seeding final_norm from backbone")
            scorer.hf_model.model.final_norm.load_state_dict(
                full_backbone.model.final_norm.state_dict()
            )

        # 6. Classifier / Head
        if hasattr(full_backbone, "head"):
            logger.info("Seeding head from backbone")
            if hasattr(scorer.hf_model, "head"):
                scorer.hf_model.head.load_state_dict(full_backbone.head.state_dict())
            elif hasattr(scorer.hf_model, "classifier") and hasattr(
                full_backbone.head, "dense"
            ):
                # Copy head dense weight if classifier layer matches
                pass

        logger.info("Successfully initialized ModernBertMiceScorer weights.")


class ModernBertMiceScorer(AbstractModuleScorer):
    """
    Experimaestro wrapper for ModernBERT MICE V2 Architecture.
    """

    hf_id: Param[str]
    """Hugging Face checkpoint identifier"""

    tokenizer: Param[MICEQueryDocTokenizer]
    """The tokenizer for independent Q/D processing"""

    n_contextualization_layers: Param[int] = field(default=6, ignore_default=True)
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Number of bottom encoder layers for the document"""

    n_interaction_layers: Param[Optional[int]] = field(
        default=None, ignore_default=True
    )
    """Number of top encoder layers with cross-attention"""

    bound_bottom_layers: Param[bool] = field(default=True, ignore_default=True)
    """Whether to bound query and document encoding layers"""

    attn_implementation: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    _version: Constant[int] = 2

    hf_model: nn.Module

    def __initialize__(self):
        super().__initialize__()
        logger.info("Initializing ModernBertMiceCrossEncoder")
        self.tokenizer.initialize()

        # Instantiate model skeleton
        backbone_config = AutoConfig.from_pretrained(self.hf_id)
        config_kwargs = backbone_config.to_dict()
        if self.attn_implementation is not None:
            config_kwargs["_attn_implementation"] = self.attn_implementation

        mice_config = ModernBertMiceConfig(
            n_contextualization_layers=self.n_contextualization_layers,
            n_docs_ctx_layers=self.n_docs_ctx_layers,
            n_interaction_layers=self.n_interaction_layers,
            bound_bottom_layers=self.bound_bottom_layers,
            **config_kwargs,
        )

        hf_model = ModernBertMiceForSequenceClassification(mice_config)
        self.add_module("hf_model", hf_model)

        attn_impl = getattr(hf_model.config, "_attn_implementation", "sdpa")
        if "flash" in str(attn_impl).lower():
            logger.info(
                f"Using Flash Attention 2 for MICE V2 ModernBERT (`{attn_impl}`)."
            )
            logger.warning(
                "FlashAttention only supports fp16 and bf16 data types. "
                "Ensure your learner config specifies `fabric.precision: bf16-mixed` (or 16-mixed)."
            )
        else:
            logger.warning(
                f"Flash Attention 2 is not enabled or available (using `{attn_impl}`). "
                "For optimal performance and unpadded attention, set `_attn_implementation='flash_attention_2'`."
            )

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options=None,
        input_nature: QueryDocInput = QueryDocInput.PAIRS,
    ) -> MICETokenizedTexts:
        return self.tokenizer.tokenize(
            input_records, options=options, input_nature=input_nature
        )

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[MICETokenizedTexts] = None,
    ) -> torch.Tensor:
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        if tokenized is None:
            tokenized = self.batch_tokenize(inputs, input_nature=QueryDocInput.PAIRS)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)

        out = self.hf_model(
            query_ids=tokenized_q.ids,
            query_mask=tokenized_q.mask,
            doc_ids=tokenized_docs.ids,
            doc_mask=tokenized_docs.mask,
        )

        return out.logits.squeeze(-1)

    def export_to_st(self, save_directory: str):
        """Export model in Sentence Transformers compatible format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.hf_model.save_pretrained(save_path)
        if hasattr(self.tokenizer, "tokenizer"):
            self.tokenizer.tokenizer.save_pretrained(save_path)
        logger.info(f"Model exported for Sentence Transformers at {save_path}")

    def save_model(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.hf_model.state_dict(), path / "pytorch_model.bin")
        self.hf_model.config.save_pretrained(path)

    def load_model(self, path: Path):
        config = ModernBertMiceConfig.from_pretrained(path)
        hf_model = ModernBertMiceForSequenceClassification(config)
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")
        hf_model.load_state_dict(state_dict)
        self.add_module("hf_model", hf_model)

    def loader_config(self, path: Path, *, settings=None) -> "SimpleModuleLoader":
        return SimpleModuleLoader.C(value=self, path=path, settings=settings)


def mice_v2_scorer(
    hf_id: str = "jhu-clsp/ettin-encoder-17m",
    n_contextualization_layers: int = 4,
    n_docs_ctx_layers: Optional[int] = None,
    n_interaction_layers: Optional[int] = None,
    bound_bottom_layers: bool = True,
    attn_implementation: Optional[str] = None,
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
    max_length: int = 512,
) -> Tuple[ModernBertMiceScorer, List[LightweightTask]]:
    """
    Factory function for ModernBERT MICE V2 scorer.
    """
    tokenizer = MICEQueryDocTokenizer.C(
        model_id=hf_id,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length,
        max_length=max_length,
    )

    scorer = ModernBertMiceScorer.C(
        hf_id=hf_id,
        tokenizer=tokenizer,
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        n_interaction_layers=n_interaction_layers,
        bound_bottom_layers=bound_bottom_layers,
        attn_implementation=attn_implementation,
    )

    init_task = InitMICEV2ModernBERTFromHFID.C(scorer=scorer)

    return scorer, [init_task]
