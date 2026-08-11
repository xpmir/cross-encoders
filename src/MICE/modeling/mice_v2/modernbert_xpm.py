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
from xpm_torch.module import SimpleModuleLoader, fallback_fa2_if_incompatible_precision
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
        logger.info(f"Building MICE from ModernBERT Encoder {scorer.hf_id}")
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
            logger.info(
                f"Seeding {scorer.n_contextualization_layers} bottom layers from backbone"
            )
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

    _version: Constant[str] = "2.0.1"

    hf_model: nn.Module

    def __initialize__(self):
        super().__initialize__()
        logger.info("Initializing ModernBertMiceCrossEncoder")
        self.tokenizer.initialize()

        # Instantiate model config (MICE config if already exported, or backbone config skeleton)
        loaded_config = AutoConfig.from_pretrained(self.hf_id)
        if (
            isinstance(loaded_config, ModernBertMiceConfig)
            or getattr(loaded_config, "model_type", None) == "modernbert_mice"
        ):
            logger.info(f"Loaded existing MICE config from {self.hf_id}")
            mice_config = loaded_config
            if self.attn_implementation is not None:
                mice_config._attn_implementation = self.attn_implementation
        else:
            logger.info(f"Building ModernBertMiceConfig from backbone {self.hf_id}")
            config_kwargs = loaded_config.to_dict()
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
        else:
            logger.warning(
                f"Flash Attention 2 is not enabled or available (using `{attn_impl}`). "
                "For optimal performance and unpadded attention, set `_attn_implementation='flash_attention_2'`."
            )

    def setup_with_fabric(self, fabric) -> torch.nn.Module:
        """Sets up the module with PyTorch Lightning Fabric, checking precision compatibility.

        If Fabric runs in float32 precision (precision is None or '32-true'), FlashAttention-2
        cannot execute and is automatically downgraded to 'sdpa'.

        Args:
            fabric: The PyTorch Lightning Fabric instance.

        Returns:
            torch.nn.Module: The Fabric-wrapped module proxy.
        """
        # extra check for FlashAttention-2 compatibility and fallback to SDPA if needed.
        fallback_fa2_if_incompatible_precision(self.hf_model, fabric)
        return super().setup_with_fabric(fabric)

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

    def write_hub_extras(self, save_directory: Path):
        """Called by TorchHFHub during model export."""
        self.export_to_st(save_directory)

    def export_to_st(
        self,
        save_directory: str,
        repo_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Export model in Sentence Transformers compatible format, including model card."""
        from xpm_torch.utils import build_model_card
        from safetensors.torch import save_file

        save_path = Path(save_directory)

        # Handle invocation on Config object
        if hasattr(self, "instance") and not getattr(self, "_initialized", False):
            inst = self.instance()
            inst.initialize()
            inst.export_to_st(save_directory, repo_id=repo_id, metadata=metadata)
            return

        bin_path = save_path / "pytorch_model.bin"
        st_path = save_path / "model.safetensors"

        # Recovery for legacy folder model.safetensors/ if present from previous cluster runs
        if st_path.is_dir():
            import shutil

            logger.info(f"Resolving legacy directory structure at {st_path}...")
            inner_bin = st_path / "pytorch_model.bin"
            inner_st = st_path / "model.safetensors"
            tmp_st = save_path / "model.safetensors.tmp"

            if inner_st.is_file():
                shutil.move(inner_st, tmp_st)
            elif inner_bin.is_file():
                weights = torch.load(inner_bin, map_location="cpu", weights_only=True)
                save_file(weights, tmp_st)

            shutil.rmtree(st_path)
            shutil.move(tmp_st, st_path)

        # Convert existing legacy pytorch_model.bin to model.safetensors file if needed
        if bin_path.is_file() and not st_path.is_file():
            logger.info(f"Converting legacy {bin_path} to {st_path}...")
            weights = torch.load(bin_path, map_location="cpu", weights_only=True)
            save_file(weights, st_path)
            bin_path.unlink(missing_ok=True)

        # Save model and tokenizer files if model.safetensors is not present
        if not st_path.is_file():
            self.save_model(save_path)
        else:
            if not (save_path / "config.json").exists() and hasattr(self, "hf_model"):
                self.hf_model.config.save_pretrained(save_path)
            if hasattr(self, "tokenizer"):
                if (
                    hasattr(self.tokenizer, "tokenizer")
                    and self.tokenizer.tokenizer is not None
                ):
                    self.tokenizer.tokenizer.save_pretrained(save_path)
                elif hasattr(self.tokenizer, "save_pretrained"):
                    self.tokenizer.save_pretrained(save_path)

        # Generate and write README.md model card using xpm_torch.utils.build_model_card
        card_template_path = Path(__file__).parent / "Micev2Card.md"
        metadata = metadata or {}
        base_name = getattr(self, "hf_id", "ModernBERT")
        model_name = repo_id or save_path.name

        cfg = (
            getattr(self.hf_model, "config", None)
            if hasattr(self, "hf_model")
            else None
        )
        n_ctx = getattr(
            cfg,
            "n_contextualization_layers",
            getattr(self, "n_contextualization_layers", 2),
        )
        n_inter = getattr(
            cfg,
            "n_interaction_layers",
            getattr(self, "n_interaction_layers", None),
        )
        if n_inter is None and cfg is not None:
            n_inter = getattr(cfg, "num_hidden_layers", 0) - n_ctx

        build_model_card(
            save_path=save_path,
            template_path=card_template_path,
            model_name=model_name,
            base_model=base_name,
            n_contextualization_layers=n_ctx,
            n_interaction_layers=n_inter,
            loss=metadata.get("loss", "not specified"),
            k=metadata.get("k", "not specified"),
            retriever=metadata.get("retriever", "not specified"),
            results=metadata.get("results", "Evaluations pending."),
        )

        logger.info(f"Model exported for Sentence Transformers at {save_path}")

    def save_model(self, path: Path):
        """Save the model architecture config, state dict weights, and tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Disconnect base model remote reference to prevent HF Hub network requests on reload
        model_name_id = getattr(self, "hf_id", path.name)
        self.hf_model.config._name_or_path = model_name_id
        if hasattr(self.hf_model.config, "base_model_name_or_path"):
            self.hf_model.config.base_model_name_or_path = model_name_id

        # 1. Save HF model (config.json + model.safetensors file)
        self.hf_model.save_pretrained(path, safe_serialization=True)
        logger.info(f"Saved ModernBERT MICE model configuration and weights to {path}")

        # 2. Save tokenizer
        if (
            hasattr(self.tokenizer, "tokenizer")
            and self.tokenizer.tokenizer is not None
        ):
            self.tokenizer.tokenizer.name_or_path = str(path)
            self.tokenizer.tokenizer.save_pretrained(path)
            logger.info(f"Saved tokenizer to {path}")
        elif hasattr(self.tokenizer, "save_pretrained"):
            if hasattr(self.tokenizer, "name_or_path"):
                self.tokenizer.name_or_path = str(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Saved tokenizer to {path}")
        else:
            logger.warning(
                f"Could not save tokenizer to {path}: no `save_pretrained` method found on tokenizer {type(self.tokenizer)}"
            )

    def load_model(self, path: Path):
        """Load model architecture config, state dict weights, and tokenizer from directory."""
        path = Path(path)
        if path.is_file():
            # Always try to load from the parent directory if a file is provided
            path = path.parent

        if not path.exists():
            raise FileNotFoundError(f"Model directory '{path}' does not exist.")

        if not self._initialized:
            if (path / "config.json").exists():
                logger.info(f"Loading MICE config from {path / 'config.json'}")
                mice_config = AutoConfig.from_pretrained(path)
                if self.attn_implementation is not None:
                    mice_config._attn_implementation = self.attn_implementation
                hf_model = ModernBertMiceForSequenceClassification(mice_config)
                self.add_module("hf_model", hf_model)
                self._initialized = True
            else:
                self.initialize()

        if (path / "model.safetensors").exists():
            from safetensors.torch import load_file

            state_dict = load_file(path / "model.safetensors")
            self.hf_model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {path / 'model.safetensors'}")
        elif (path / "pytorch_model.bin").exists():
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")
            self.hf_model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {path / 'pytorch_model.bin'}")
        else:
            super().load_model(path)

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
