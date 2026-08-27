"""
Experimaestro-IR integration for ModernBERT MICE V2 architecture.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import logging
import shutil
import json
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from experimaestro import LightweightTask, Param, Meta, Constant, field
from datamaestro_ir.data.base import IDTextRecord
from xpmir.text.tokenizers import TokenizerOptions
from xpmir.rankers.scorer import AbstractModuleScorer
from xpmir.letor.records import BaseItems
from xpmir.text.encoders import (
    TextEncoderBase,
    EncoderOutput,
    TokensEncoderOutput,
    TokensRepresentationOutput,
)
from xpm_torch.utils import to_device
from xpm_torch.trainers import TrainingHook, TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.metrics import ScalarMetric
from xpm_torch.module import (
    ModuleLoader,
    SimpleModuleLoader,
    ReadmeSection,
    fallback_fa2_if_incompatible_precision,
)
from MICE.modeling.mice import MICEQueryDocTokenizer, MICETokenizedTexts, QueryDocInput
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)


logger = logging.getLogger(__name__)


class UtilityRegularizer(TrainingHook):
    """Training hook for computing token utility L1 regularization loss and sparsity metrics"""

    lambda_reg: Param[float] = field(default=0.1)
    """Regularization weight for utility predictor L1 loss"""

    sparsity_threshold: Meta[float] = field(default=0.5)
    """Threshold below which a token utility score is considered suppressed / sparse (default 0.5) affects sparsity metric computation only"""

    histogram_interval: Meta[int] = field(default=0)
    """Interval in steps for writing utility distribution histograms to TensorBoard (0 to disable)"""

    version: Constant[str] = "1.0.0"

    def __call__(
        self,
        info: TrainerContext,
        utilities: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
    ):
        if self.lambda_reg <= 0.0 or utilities is None:
            return

        if (
            doc_mask is not None
            and utilities.ndim == 3
            and utilities.shape[0] == doc_mask.shape[0]
            and utilities.shape[1] == doc_mask.shape[1]
        ):  # we are using sdpa - need to mask tokens
            mask = doc_mask.unsqueeze(-1).bool()
            valid_utilities = utilities.masked_select(mask)
            if valid_utilities.numel() > 0:
                mean_utility = valid_utilities.mean()
                sparsity = (valid_utilities < self.sparsity_threshold).float().mean()
            else:
                mean_utility = utilities.mean()
                sparsity = (utilities < self.sparsity_threshold).float().mean()
        else:
            # under FA2, padding tokens are already removed, so we can just take the mean over all tokens
            mean_utility = utilities.mean()
            sparsity = (utilities < self.sparsity_threshold).float().mean()
            valid_utilities = utilities.view(-1)

        reg_loss = mean_utility
        info.add_loss(Loss("utility_reg", reg_loss, self.lambda_reg))

        if getattr(info, "metrics", None) is not None:
            nrecords = doc_mask.shape[0] if doc_mask is not None else 1
            info.metrics.add(
                ScalarMetric(
                    "utility_reg",
                    float((self.lambda_reg * mean_utility).item()),
                    nrecords,
                )
            )
            info.metrics.add(
                ScalarMetric("utility/mean", float(mean_utility.item()), nrecords)
            )
            info.metrics.add(
                ScalarMetric("utility/sparsity", float(sparsity.item()), nrecords)
            )

        # Log TensorBoard histogram at validation interval boundaries
        writer = getattr(info, "writer", None)
        if (
            self.histogram_interval > 0
            and writer is not None
            and (info.state.step % self.histogram_interval == 0)
            and valid_utilities.numel() > 0
        ):
            writer.add_histogram(
                "train/utility/distribution",
                valid_utilities.detach(),
                global_step=info.state.step,
            )


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
                    norm_to_copy = (
                        src_layer.attn_norm
                        if hasattr(src_layer, "attn_norm")
                        and isinstance(src_layer.attn_norm, nn.LayerNorm)
                        else (
                            src_layer.mlp_norm
                            if hasattr(src_layer, "mlp_norm")
                            and isinstance(src_layer.mlp_norm, nn.LayerNorm)
                            else None
                        )
                    )
                    if norm_to_copy is not None:
                        top_layer.cross_attn_norm.load_state_dict(
                            norm_to_copy.state_dict()
                        )
                        top_layer.self_attn_norm.load_state_dict(
                            norm_to_copy.state_dict()
                        )
                        if isinstance(top_layer.doc_cross_attn_norm, nn.LayerNorm):
                            top_layer.doc_cross_attn_norm.load_state_dict(
                                norm_to_copy.state_dict()
                            )

                    if hasattr(src_layer, "mlp_norm") and isinstance(
                        src_layer.mlp_norm, nn.LayerNorm
                    ):
                        top_layer.mlp_norm.load_state_dict(
                            src_layer.mlp_norm.state_dict()
                        )

                    top_layer.self_attn.load_state_dict(src_layer.attn.state_dict())
                    top_layer.cross_attn.Wqkv.load_state_dict(
                        src_layer.attn.Wqkv.state_dict()
                    )
                    top_layer.cross_attn.Wo.load_state_dict(
                        src_layer.attn.Wo.state_dict()
                    )
                    top_layer.mlp.load_state_dict(src_layer.mlp.state_dict())

        # 4. Rotary embeddings
        logger.info("Seeding rotary_emb from backbone")
        scorer.hf_model.model.rotary_emb.load_state_dict(
            full_backbone.model.rotary_emb.state_dict()
        )

        # 5. Final LayerNorm
        if getattr(scorer.hf_model.config, "bottom_final_norm", True):
            logger.info("Seeding bottom final_norm from backbone")
            scorer.hf_model.model.final_norm.load_state_dict(
                full_backbone.model.final_norm.state_dict()
            )

        if (
            hasattr(scorer.hf_model, "top_final_norm")
            and scorer.hf_model.top_final_norm is not None
        ):
            logger.info("Seeding top_final_norm from backbone")
            scorer.hf_model.top_final_norm.load_state_dict(
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


def find_latest_epoch_dir(ckpt_path: Path) -> Optional[Path]:
    """Finds the latest epoch checkpoint directory (e.g. epoch-00000150) containing trainer.pth & optimizer.pth."""
    search_dirs = []
    checkpoints_dir = next(
        (p for p in [ckpt_path, *ckpt_path.parents] if p.name == "checkpoints"),
        None,
    )
    if checkpoints_dir is not None:
        search_dirs.append(checkpoints_dir)
    search_dirs.extend([ckpt_path, ckpt_path.parent])

    for d in search_dirs:
        if not d.exists():
            continue
        epoch_dirs = []
        for f in d.glob("epoch-*"):
            if f.is_dir():
                try:
                    epoch_num = int(f.name[len("epoch-") :])
                    epoch_dirs.append((epoch_num, f))
                except ValueError:
                    pass
        if epoch_dirs:
            epoch_dirs.sort(key=lambda x: x[0], reverse=True)
            for _, epoch_dir in epoch_dirs:
                if (epoch_dir / "trainer.pth").exists() and (
                    epoch_dir / "optimizer.pth"
                ).exists():
                    return epoch_dir
    return None


def is_rank_zero() -> bool:
    """Returns True if this is global rank 0 or running in a single-process environment."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    for env_var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                pass
    return True


class InitMICEV2FromCheckpoint(LightweightTask):
    """Initializes a Phase 2 MICE V2 model from a Phase 1 trained checkpoint path/loader and freezes bottom layers & utility predictor."""

    scorer: Param["ModernBertMiceScorer"]

    checkpoint_loader: Param[ModuleLoader]
    """Path or ModuleLoader to the Phase 1 trained checkpoint (should contain model.safetensors or model/ directory)."""

    def execute(self):
        if not is_rank_zero():
            logger.info(
                "[InitMICEV2] Skipping checkpoint initialization on non-zero rank (RANK=%s, LOCAL_RANK=%s)",
                os.environ.get("RANK", "unknown"),
                os.environ.get("LOCAL_RANK", "unknown"),
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        ckpt_path = Path(self.checkpoint_loader.path)
        logger.info("[InitMICEV2] Copying Phase 1 checkpoint at: %s", ckpt_path)

        latest_epoch_dir = find_latest_epoch_dir(ckpt_path)
        assert latest_epoch_dir is not None, (
            f"[InitMICEV2] No valid epoch directory found in {ckpt_path}"
        )

        # 1. Read Phase 1 step count from info.json
        phase1_steps = 0
        if (latest_epoch_dir / "info.json").exists():
            try:
                with (latest_epoch_dir / "info.json").open("rt") as fp:
                    p1_info = json.load(fp)
                    phase1_steps = p1_info.get("steps", 0)
                logger.info(
                    "[InitMICEV2] Preserving Phase 1 step count (%d) for TensorBoard continuity at epoch 0",
                    phase1_steps,
                )
            except Exception as e:
                logger.warning(
                    "[InitMICEV2] Could not read steps from Phase 1 info.json: %s", e
                )

        # Helper function to populate epoch-00000000 directory
        def copy_to_epoch_zero(target_base_dir: Path):
            epoch_zero = target_base_dir / "epoch-00000000"
            if epoch_zero.exists():
                logging.warning(
                    "[InitMICEV2] Removing existing epoch-00000000 directory at %s",
                    epoch_zero,
                )
                shutil.rmtree(epoch_zero)

            shutil.copytree(latest_epoch_dir, epoch_zero)

            # Copy trainer & optimizer state dicts if present
            if (latest_epoch_dir / "trainer.pth").exists():
                shutil.copy2(
                    latest_epoch_dir / "trainer.pth", epoch_zero / "trainer.pth"
                )
            # Write info.json resetting epoch to 0 but preserving total steps
            with (epoch_zero / "info.json").open("wt") as fp:
                json.dump({"epoch": 0, "steps": phase1_steps}, fp)

            logger.info("[InitMICEV2] Populated epoch-00000000 at %s", epoch_zero)

        # Populate process cwd checkpoints/epoch-00000000
        copy_to_epoch_zero(Path.cwd() / "checkpoints")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


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

    utility_pred: Param[bool] = field(default=False, ignore_default=True)
    """Whether to add a 2layer MLP utility predictor applied to the document hidden states before upper layers"""

    hard_pruning: Param[bool] = field(default=False, ignore_default=True)
    """Whether to hard-prune tokens with utility < hard_prune_threshold in Phase 2 training"""

    hard_prune_threshold: Param[float] = field(default=0.5, ignore_default=True)
    """Utility threshold below which tokens are hard-pruned in Phase 2 training (default 0.5)"""

    normalize_doc_embeds: Param[bool] = field(default=False, ignore_default=True)
    """Whether to L2-normalize document hidden states before utility prediction and upper interaction layers"""

    use_upper_rope: Param[bool] = field(default=True, ignore_default=True)
    """Whether to use Rotary Position Embedding (RoPE) for query tokens in upper interaction layers (disabled for documents)"""

    top_layers_doc_ln: Param[bool] = field(default=False, ignore_default=True)
    """Whether to use a document LayerNorm in top interaction layers and multiply token utilities post-LayerNorm"""

    bottom_final_norm: Param[bool] = field(default=True, ignore_default=True)
    """Whether to apply LayerNorm at the output of bottom contextualization layers"""

    top_final_norm: Param[bool] = field(default=False, ignore_default=True)
    """Whether to apply LayerNorm at the output of top interaction layers before classifier"""

    _version: Constant[str] = "2.0.2"

    hf_model: nn.Module

    def freeze_bottom_and_utility(self):
        """Freezes bottom layers, embeddings, and utility predictor."""
        if hasattr(self, "hf_model") and hasattr(
            self.hf_model, "freeze_bottom_and_utility"
        ):
            self.hf_model.freeze_bottom_and_utility()

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
            if self.utility_pred:
                mice_config.utility_pred = True
            if self.hard_pruning:
                mice_config.hard_pruning = True
                mice_config.hard_prune_threshold = self.hard_prune_threshold
            if self.normalize_doc_embeds:
                mice_config.normalize_doc_embeds = True
            if not self.use_upper_rope:
                mice_config.use_upper_rope = False
            if self.top_layers_doc_ln:
                mice_config.top_layers_doc_ln = True
            if not self.bottom_final_norm:
                mice_config.bottom_final_norm = False
            if self.top_final_norm:
                mice_config.top_final_norm = True
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
                utility_pred=self.utility_pred,
                hard_pruning=self.hard_pruning,
                hard_prune_threshold=self.hard_prune_threshold,
                normalize_doc_embeds=self.normalize_doc_embeds,
                use_upper_rope=self.use_upper_rope,
                top_layers_doc_ln=self.top_layers_doc_ln,
                bottom_final_norm=self.bottom_final_norm,
                top_final_norm=self.top_final_norm,
                **config_kwargs,
            )

        hf_model = ModernBertMiceForSequenceClassification(mice_config)
        self.add_module("hf_model", hf_model)

        if self.hard_pruning:
            self.freeze_bottom_and_utility()

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

    def encode_documents(
        self, doc_ids: torch.Tensor, doc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode documents through bottom contextualization layers."""
        return self.hf_model.encode_documents(doc_ids=doc_ids, doc_mask=doc_mask)

    def get_document_encoder(self) -> TextEncoderBase:
        """Returns a document encoder for fast-plaid index building."""
        model_cfg = getattr(self, "config", self)
        return MiceV2DocumentEncoder.C(model=model_cfg)

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
        doc_hidden_states: Optional[List[torch.Tensor]] = None,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        if tokenized is None:
            tokenized = self.batch_tokenize(inputs, input_nature=QueryDocInput.PAIRS)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)

        if doc_hidden_states is not None:
            # Decompressed document token embeddings from fast-plaid index (list of 2D tensors [doc_len, dim])
            doc_tensors = [to_device(d, self.device) for d in doc_hidden_states]
            max_d_len = max(d.shape[0] for d in doc_tensors)
            batch_size = len(doc_tensors)
            dim = doc_tensors[0].shape[-1]

            doc_hidden = torch.zeros(
                (batch_size, max_d_len, dim),
                dtype=doc_tensors[0].dtype,
                device=self.device,
            )
            doc_mask = torch.zeros(
                (batch_size, max_d_len),
                dtype=torch.long,
                device=self.device,
            )
            for i, d in enumerate(doc_tensors):
                seq_len = d.shape[0]
                doc_hidden[i, :seq_len] = d
                doc_mask[i, :seq_len] = 1

            out = self.hf_model(
                query_ids=tokenized_q.ids,
                query_mask=tokenized_q.mask,
                doc_hidden=doc_hidden,
                doc_mask=doc_mask,
            )
            target_doc_mask = doc_mask
        else:
            tokenized_docs = to_device(tokenized.tokenized_docs, self.device)
            out = self.hf_model(
                query_ids=tokenized_q.ids,
                query_mask=tokenized_q.mask,
                doc_ids=tokenized_docs.ids,
                doc_mask=tokenized_docs.mask,
            )
            target_doc_mask = tokenized_docs.mask

        if (
            info is not None
            and getattr(self.hf_model, "last_utilities", None) is not None
        ):
            for hook in info.hooks(UtilityRegularizer):
                hook(info, self.hf_model.last_utilities, target_doc_mask)

        return out.logits.squeeze(-1)

    def hub_readme_sections(self) -> List["ReadmeSection"]:
        """Returns structured README sections (main card template + separate evaluations section)."""
        from xpm_torch.module import ReadmeSection
        from xpm_torch.utils import count_safetensors_params, format_num_params
        from huggingface_hub import ModelCard, ModelCardData

        sections = []
        template_path = Path(__file__).parent / "Micev2Card.md"

        export_dir = getattr(self, "_last_export_dir", None) or getattr(
            getattr(self, "__config__", None), "_last_export_dir", None
        )
        total_params_str = "N/A"
        if export_dir:
            st_path = Path(export_dir) / "model.safetensors"
            if st_path.is_file():
                try:
                    num_params = count_safetensors_params(st_path)
                    total_params_str = (
                        f"{num_params:,} ({format_num_params(num_params)})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute parameter count from safetensors: {e}"
                    )

        metadata = (
            getattr(self, "_export_metadata", None)
            or getattr(getattr(self, "__config__", None), "_export_metadata", {})
            or {}
        )
        base_name = getattr(self, "hf_id", "ModernBERT")
        model_name = (
            getattr(self, "model_name", None)
            or getattr(getattr(self, "__config__", None), "model_name", None)
            or (Path(export_dir).name if export_dir else "MICE v2 Model")
        )

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

        # 1. Main Model Card Section (from Micev2Card.md template)
        if template_path.exists():
            card_data = ModelCardData(
                language="en",
                license="apache-2.0",
                base_model=str(base_name),
                model_name=str(model_name),
                pipeline_tag="text-classification",
            )
            card = ModelCard.from_template(
                card_data,
                template_path=str(template_path),
                base=str(base_name),
                model_id=str(model_name),
                total_parameters=total_params_str,
                n_contextualization_layers=n_ctx,
                n_interaction_layers=n_inter,
                loss=metadata.get("loss", "N/A"),
            )
            sections.append(ReadmeSection(key="model_card", content=card.content))

        # 2. Separate Evaluations Section (positioned after model_card)
        results_md = metadata.get("results")
        if results_md:
            retriever = metadata.get("retriever", "N/A")
            k = metadata.get("k", "N/A")
            eval_content = f"## Evaluations\n\nWe provide evaluations of this cross-encoder re-ranking top `{k}` documents retrieved by `{retriever}`:\n\n{results_md}"
            sections.append(
                ReadmeSection(
                    key="evaluations",
                    after="model_card",
                    content=eval_content,
                )
            )

        return sections

    def write_hub_extras(self, save_directory: Path):
        """Called by TorchHFHub during model export."""
        save_path = Path(save_directory)
        self._last_export_dir = save_path
        self.save_model(save_path)

    def _write_st_metadata(self, save_directory: Path):
        """Write native SentenceTransformers metadata files for CrossEncoder compatibility."""
        import json

        save_directory = Path(save_directory)

        # 1. modules.json
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.base.modules.transformer.Transformer",
            }
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # 2. config_sentence_transformers.json
        st_config = {
            "activation_fn": "torch.nn.modules.linear.Identity",
            "default_prompt_name": None,
            "model_type": "CrossEncoder",
            "prompts": {},
        }
        (save_directory / "config_sentence_transformers.json").write_text(
            json.dumps(st_config, indent=2)
        )

        # 3. sentence_bert_config.json
        sb_config = {
            "transformer_task": "sequence-classification",
            "modality_config": {
                "text": {
                    "method": "forward",
                    "method_output_name": "logits",
                }
            },
            "module_output_name": "scores",
        }
        (save_directory / "sentence_bert_config.json").write_text(
            json.dumps(sb_config, indent=2)
        )

    def export_to_st(
        self,
        save_directory: str,
        repo_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Export model in Sentence Transformers compatible format, including model card."""
        from xpm_torch.module import assemble_readme_sections

        save_path = Path(save_directory)
        self._last_export_dir = save_path
        if repo_id:
            self.model_name = repo_id
        if metadata:
            self._export_metadata = metadata

        self.write_hub_extras(save_path)
        self._write_st_metadata(save_path)
        sections = self.hub_readme_sections()
        if sections:
            readme = assemble_readme_sections([], sections)
            (save_path / "README.md").write_text(readme)

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

        if (path / "model" / "model.safetensors").exists() or (
            path / "model" / "pytorch_model.bin"
        ).exists():
            path = path / "model"

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


class MiceV2DocumentEncoder(TextEncoderBase):
    """Document encoder for ModernBERT MICE v2, interfacing with PlaidIndexBuilder."""

    model: Param[ModernBertMiceScorer]

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def dimension(self):
        return getattr(self.model.hf_model.config, "hidden_size", 768)

    @staticmethod
    def _token_mask(output: TokensRepresentationOutput) -> Optional[torch.Tensor]:
        mask = output.tokenized.mask
        if mask is None:
            return None
        return mask.to(output.value.device).bool()

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return per-token embeddings filtering padding out."""
        output = self.encode_documents(records)
        mask = self._token_mask(output)
        value = output.value
        if mask is None:
            return [value[i] for i in range(value.shape[0])]
        return [value[i][mask[i]] for i in range(value.shape[0])]

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        options = TokenizerOptions(max_length=self.model.tokenizer.max_doc_length)
        output = self(records, options=options)
        return output

    def forward(
        self,
        inputs: List[IDTextRecord],
        *args,
        options: Optional[TokenizerOptions] = None,
    ) -> EncoderOutput:
        assert len(args) == 0, "Unhandled extra arguments"
        tokenized = self.model.batch_tokenize(
            inputs, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.model.device:
            tokenized = tokenized.to(self.model.device)
        encoded = self.model.encode_documents(tokenized.ids, tokenized.mask)
        return TokensEncoderOutput(tokenized, encoded)


def mice_v2_scorer(
    hf_id: str = "jhu-clsp/ettin-encoder-17m",
    n_contextualization_layers: int = 4,
    n_docs_ctx_layers: Optional[int] = None,
    n_interaction_layers: Optional[int] = None,
    bound_bottom_layers: bool = True,
    attn_implementation: Optional[str] = None,
    utility_pred: bool = False,
    hard_pruning: bool = False,
    hard_prune_threshold: float = 0.5,
    normalize_doc_embeds: bool = False,
    use_upper_rope: bool = True,
    top_layers_doc_ln: bool = False,
    bottom_final_norm: bool = True,
    top_final_norm: bool = False,
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
        utility_pred=utility_pred,
        hard_pruning=hard_pruning,
        hard_prune_threshold=hard_prune_threshold,
        normalize_doc_embeds=normalize_doc_embeds,
        use_upper_rope=use_upper_rope,
        top_layers_doc_ln=top_layers_doc_ln,
        bottom_final_norm=bottom_final_norm,
        top_final_norm=top_final_norm,
    )

    init_task = InitMICEV2ModernBERTFromHFID.C(scorer=scorer)

    return scorer, [init_task]
