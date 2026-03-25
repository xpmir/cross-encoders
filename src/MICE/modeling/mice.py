from typing import List, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn
import logging

from experimaestro import Param, LightweightTask, Constant
from xpmir.text import TokenizedTexts
from xpmir.letor.records import BaseItems
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.utils import to_device

from xpmir.text.huggingface.tokenizers import HFTokenizer
from xpmir.text.tokenizers import TokenizerOptions

# Configuration and common types

# Transformers imports with safety checks
try:
    from transformers import (
        ModernBertConfig,
        AutoModel,
        AutoConfig,
        AutoModelForSequenceClassification,
    )
    from transformers.models.bert.modeling_bert import BertLayer
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertMLP,
        ModernBertAttention,
    )
    from transformers.models.modernbert_decoder.modeling_modernbert_decoder import (
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

logger = logging.getLogger(__name__)


class MICETokenizedTexts(NamedTuple):
    """Container for MICE tokenized inputs (separate query and document streams)"""

    tokenized_q: TokenizedTexts
    """tokenized Queries"""

    tokenized_docs: TokenizedTexts
    """tokenized Documents"""


class MICEQueryDocTokenizer(HFTokenizer):
    """Specific tokenizer for MICE that handles query and document independently"""

    max_query_length: Param[Optional[int]]
    """maximum number of tokens for the query side"""

    max_doc_length: Param[Optional[int]]
    """maximum number of tokens for the document side"""

    def __post_init__(self):
        super().__post_init__()
        if self.max_doc_length is None:
            self.max_doc_length = self.max_length
        if self.max_query_length is None:
            self.max_query_length = self.max_length

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> MICETokenizedTexts:
        # Determine per-side token limits
        q_max = self.max_query_length
        d_max = self.max_doc_length

        ix_qs, ix_ds = input_records.pairs()
        queries = [input_records.unique_topics[i]["text_item"].text for i in ix_qs]
        docs = [input_records.unique_documents[i]["text_item"].text for i in ix_ds]

        def _encode(texts: List[str], max_tokens: int):
            r = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens,
                padding=True,
                return_tensors="pt",
                return_length=True,
            )
            return TokenizedTexts(
                tokens=None,
                ids=r["input_ids"],
                lens=r["length"].tolist(),
                mask=r.get("attention_mask", None),
                token_type_ids=r.get("token_type_ids", None),
            )

        return MICETokenizedTexts(
            tokenized_q=_encode(queries, q_max),
            tokenized_docs=_encode(docs, d_max),
        )


class MiceCrossEncoder(AbstractModuleScorer):
    """
    Mid-Fusion Cross Encoder base Architecture, with Cross-Attention in the top layers.
    The bottom layers encode query and document independently, while the top layers
    allow the query to attend to the document via cross-attention mechanisms.
    """

    ## Parameters for Config

    hf_id: Param[str]
    """Hugging Face checkpoint identifier that provides weights and config. Must be a BERT-like model."""

    tokenizer: Param[MICEQueryDocTokenizer]
    """The tokenizer for independent Q/D processing"""

    merge_layer: Param[int] = 6
    """Mid-fusion index: encoder layers are split into bottom (independent) and top (cross-attention)"""

    drop_layer: Param[int] = 0
    """Layer at which to drop backbone layers"""

    mask_cls_to_doc: Param[bool] = True
    """Whether to mask the [CLS] token from attending to document tokens."""

    mask_query_to_cls: Param[bool] = True
    """Whether to mask query tokens from attending to the [CLS] token (using it as a sink)"""

    freeze_base: Param[bool] = False
    """Whether to freeze the bottom layers during finetuning"""

    random_top_layers: Param[bool] = False
    """Whether to initialize top layers randomly instead of copying from backbone"""

    compress_dim: Param[float] = 1.0
    """Factor by which to divide the hidden dimensions of the top layers"""

    _version: Constant[int] = 2
    """Model version"""

    ## Attributes (not Parameters)

    embeddings: nn.Module
    """Shared embeddings for query and document"""

    bottom_layers: nn.ModuleList
    """Bottom layers: independent encoding"""

    top_layers: nn.ModuleList
    """Top layers: cross-attention encoding"""

    def __initialize__(self):
        super().__initialize__()
        self.tokenizer.initialize()

        # Ensure _attn_implementation is not None to avoid warnings
        # Configs should be set by InitTask or manually before calling initialize()
        if hasattr(self, "config") and self.config is not None:
            if (
                not hasattr(self.config, "_attn_implementation")
                or self.config._attn_implementation is None
            ):
                self.config._attn_implementation = "eager"

        if hasattr(self, "head_config") and self.head_config is not None:
            if (
                not hasattr(self.head_config, "_attn_implementation")
                or self.head_config._attn_implementation is None
            ):
                self.head_config._attn_implementation = getattr(
                    self.config, "_attn_implementation", "eager"
                )

    def batch_tokenize(
        self, input_records: BaseItems, options=None
    ) -> MICETokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        return self.tokenizer.tokenize(input_records, options=options)

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def get_extended_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        inverted_mask = 1.0 - mask[:, None, None, :]
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

    def get_self_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        mask_bool = mask.to(torch.bool)
        # Build a square attention map where both query and key positions must be valid tokens
        valid_pairs = mask_bool[:, None, :, None] & mask_bool[:, None, None, :]
        if self.mask_query_to_cls:
            # Additionally mask out all query tokens from attending to the [CLS] token
            valid_pairs[:, :, 1:, 0] = False
        attn_mask = torch.zeros(valid_pairs.shape, dtype=dtype, device=mask.device)
        attn_mask.masked_fill_(~valid_pairs, torch.finfo(dtype).min)

        return attn_mask

    def get_cross_attention_mask(self, query_mask, doc_mask, dtype):
        """Helper to create the -inf mask for cross-attention with rectangular support"""
        q_valid = query_mask.to(torch.bool)[:, :, None]
        d_valid = doc_mask.to(torch.bool)[:, None, :]
        valid_pairs = q_valid & d_valid
        if self.mask_cls_to_doc:
            # Additionally mask out the [CLS] token from attending to document tokens
            valid_pairs[:, 0, :] = False
        attn_mask = torch.zeros(
            (valid_pairs.size(0), 1, valid_pairs.size(1), valid_pairs.size(2)),
            dtype=dtype,
            device=query_mask.device,
        )
        attn_mask.masked_fill_(~valid_pairs[:, None, :, :], torch.finfo(dtype).min)
        return attn_mask


class BertMiceCrossEncoder(MiceCrossEncoder):
    """Mid-Fusion Cross Encoder based on BERT Architecture."""

    def __initialize__(self):
        super().__initialize__()
        self.head_config.is_decoder = True
        self.head_config.add_cross_attention = True

        if self.compress_dim > 1:
            self.head_config.hidden_size = int(
                self.head_config.hidden_size / self.compress_dim
            )
            self.head_config.intermediate_size = int(
                self.head_config.intermediate_size / self.compress_dim
            )
            self.head_config.num_attention_heads = int(
                self.head_config.num_attention_heads / self.compress_dim
            )
            self.adapter = nn.Linear(
                self.config.hidden_size, self.head_config.hidden_size
            )
        else:
            self.adapter = None

        # Build skeleton (initially random)
        # Note: We use the config to build the structure. Weights copied later by InitTask.
        temp_model = AutoModel.from_config(self.config)
        self.add_module("embeddings", temp_model.embeddings)
        self.add_module(
            "bottom_layers",
            nn.ModuleList([BertLayer(self.config) for _ in range(self.merge_layer)]),
        )

        num_top = (self.drop_layer or len(temp_model.encoder.layer)) - self.merge_layer
        self.add_module(
            "top_layers",
            nn.ModuleList([BertLayer(self.head_config) for _ in range(num_top)]),
        )

        self.pooler = getattr(temp_model, "pooler", None)
        if self.pooler:
            self.add_module("pooler", self.pooler)
        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.head_config.hidden_size, 1)

    def forward_bottom(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        # Compute position ids and pass them to ModernBertEncoderLayer which expects them
        batch, seq_len = input_ids.size()

        for layer in self.bottom_layers:
            x = layer(x, ext_mask)
        return x

    def forward(
        self,
        inputs: BaseItems,
        tokenized: Optional[MICETokenizedTexts] = None,
        doc_hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the Mid-Fusion Cross Encoder.
        inputs: BaseRecords containing 'topics' and 'documents' with TextItems.
        tokenized_queries: Optional pre-tokenized queries to skip tokenization step.
        tokenized_docs: Optional pre-tokenized documents to skip tokenization step.
        doc_hidden_states: Optional pre-computed document hidden states from bottom layers.
        info: TrainerContext for additional context (not used here).
        """

        # Prepare inputs
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)

        query_ids = tokenized_q.ids
        query_mask = tokenized_q.mask
        doc_ids = tokenized_docs.ids
        doc_mask = tokenized_docs.mask

        # 1. Process Query through Bottom Layers
        q_hidden = self.forward_bottom(query_ids, query_mask)

        # Mask for Self-Attention (Query) shape [batch, 1, seq_len_query, seq_len_query]
        q_ext_mask = self.get_self_attention_mask(query_mask, q_hidden.dtype)

        if doc_hidden_states is None:
            # Process Doc through Bottom Layers
            doc_hidden_states = self.forward_bottom(
                doc_ids, doc_mask
            )  # shape [batch, seq_len_doc, dim]
            # 2. Prepare Masks for Top Layers

        # Mask for Cross-Attention (Query attending to Doc) shape [batch, 1, seq_len_query, seq_len_doc]
        d_ext_mask = self.get_cross_attention_mask(query_mask, doc_mask, q_hidden.dtype)

        # 3. Process Query through Top Layers (with Cross-Attention to Doc)
        if self.adapter is not None:
            q_hidden = self.adapter(q_hidden)
            doc_hidden_states = self.adapter(doc_hidden_states)

        for layer in self.top_layers:
            # BertLayer with is_decoder=True accepts:
            # (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
            layer_out = layer(
                hidden_states=q_hidden,  # Query (Self-Attn)
                attention_mask=q_ext_mask,
                encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                encoder_attention_mask=d_ext_mask,
            )
            q_hidden = layer_out

        # 4. Score (Use [CLS] of the Query)
        if self.pooler is not None:
            pooled = self.pooler(q_hidden)
        else:
            pooled = q_hidden[:, 0, :]

        pooled = self.dropout_layer(pooled)
        score = self.classifier(pooled)
        return score.squeeze(-1)


class ModernBertCrossAttention(nn.Module):
    """Cross-attention wrapper for ModernBERT that accepts separate query/key/value tensors."""

    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

    def forward(self, query, key, value, attention_mask, **kwargs):
        q_shape = query.shape[:-1]
        query_states = (
            self.q_proj(query).view(*q_shape, -1, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.k_proj(key).view(*key.shape[:-1], -1, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(value)
            .view(*value.shape[:-1], -1, self.head_dim)
            .transpose(1, 2)
        )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*q_shape, -1).contiguous()
        return self.out_drop(self.Wo(attn_output)), attn_weights


class ModernBertCrossAttentionLayer(nn.Module):
    """
    A ModernBERT encoder layer with added cross-attention for mid-fusion ranking.
    """

    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        )
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.crossattention = ModernBertCrossAttention(
            config=config, layer_idx=layer_id
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_ids=None,
    ):
        # Self-attn residual
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        hidden_states = hidden_states + attn_out

        # Cross-attn residual
        if encoder_hidden_states is not None:
            cross_out = self.crossattention(
                query=self.attn_norm(hidden_states),
                key=self.attn_norm(encoder_hidden_states),
                value=self.attn_norm(encoder_hidden_states),
                attention_mask=encoder_attention_mask,
            )[0]
            hidden_states = hidden_states + cross_out

        # MLP residual
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return (hidden_states,)


class ModernBertMiceCrossEncoder(MiceCrossEncoder):
    """Mid-Fusion Cross Encoder based on ModernBERT Architecture."""

    pooling_method: Param[Optional[str]] = None
    """Pooling method to use for the ModernBert based scorer: cls or mean.
    Leave it to None for models coming from the Hub, as it will be inferred from the model config."""

    def __initialize__(self):
        super().__initialize__()
        pm = self.pooling_method or getattr(self.config, "classifier_pooling", "cls")
        if pm == "cls":
            self.pooling_function = lambda x: x[:, 0]
        else:
            self.pooling_function = lambda x: x.mean(dim=1)

        # Structure setup
        # Weights copied later by InitTask
        temp_model = AutoModelForSequenceClassification.from_config(self.config)
        self.add_module("embeddings", temp_model.model.embeddings)
        self.add_module(
            "bottom_layers",
            nn.ModuleList(
                [
                    type(temp_model.model.layers[0])(self.config, i)
                    for i in range(self.merge_layer)
                ]
            ),
        )

        num_top = (self.drop_layer or len(temp_model.model.layers)) - self.merge_layer
        self.add_module(
            "top_layers",
            nn.ModuleList(
                [
                    ModernBertCrossAttentionLayer(self.config, self.merge_layer + i)
                    for i in range(num_top)
                ]
            ),
        )

        self.add_module("final_norm", temp_model.model.final_norm)
        self.add_module("head", temp_model.head)
        self.dropout_layer = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(
        self, inputs: BaseItems, tokenized: Optional[MICETokenizedTexts] = None
    ):
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)

        query_ids = tokenized_q.ids
        query_mask = tokenized_q.mask
        doc_ids = tokenized_docs.ids
        doc_mask = tokenized_docs.mask

        # Pos IDs for RoPE
        def _get_pos_ids(ids):
            b, s = ids.size()
            return torch.arange(s, device=ids.device).unsqueeze(0).expand(b, s)

        # Bottom
        q_ext = self.get_extended_attention_mask(query_mask, torch.float32)
        d_ext = self.get_extended_attention_mask(doc_mask, torch.float32)

        x_q = self.embeddings(query_ids)
        x_d = self.embeddings(doc_ids)

        q_pos = _get_pos_ids(query_ids)
        d_pos = _get_pos_ids(doc_ids)

        for layer in self.bottom_layers:
            x_q = layer(x_q, q_ext, position_ids=q_pos)[0]
            x_d = layer(x_d, d_ext, position_ids=d_pos)[0]

        # Top
        q_self = self.get_self_attention_mask(query_mask, x_q.dtype)
        cross_mask = self.get_cross_attention_mask(query_mask, doc_mask, x_q.dtype)

        for layer in self.top_layers:
            x_q = layer(
                x_q,
                attention_mask=q_self,
                encoder_hidden_states=x_d,
                encoder_attention_mask=cross_mask,
                position_ids=q_pos,
            )[0]

        x_q = self.final_norm(x_q)
        pooled = self.pooling_function(x_q)
        return self.classifier(self.dropout_layer(self.head(pooled))).squeeze(-1)


class InitMICEBERTFromHFID(LightweightTask):
    """Worker-node task to load weights into MICE BERT model"""

    model: Param[BertMiceCrossEncoder]

    def execute(self):
        # Ensure model is instantiated and initialized
        model = self.model
        hf_id = model.hf_id

        # Ensure configs are available
        if not hasattr(model, "config") or model.config is None:
            model.config = AutoConfig.from_pretrained(hf_id)

        if not hasattr(model, "head_config") or model.head_config is None:
            model.head_config = AutoConfig.from_pretrained(hf_id)
            model.head_config.is_decoder = True
            model.head_config.add_cross_attention = True

        # Build the model structure first
        model.initialize()

        logger.info(f"Building MICE from pretrained Bert model: {hf_id}")

        full_bert = AutoModel.from_pretrained(hf_id)
        model.embeddings = full_bert.embeddings

        model.top_layers = nn.ModuleList()

        # Load original top layers to copy weights from
        if model.drop_layer > 0:
            assert model.drop_layer >= model.merge_layer, (
                "drop_layer must be >= merge_layer"
            )
            assert model.drop_layer < len(full_bert.encoder.layer), (
                f"drop_layer {model.drop_layer} exceeds number of layers in the backbone: {len(full_bert.encoder.layer)}"
            )
            original_top_layers = full_bert.encoder.layer[
                model.merge_layer : model.drop_layer
            ]
            logging.info(
                f"Dropping backbone layers {model.drop_layer}-{len(full_bert.encoder.layer) - 1}"
            )
        else:
            original_top_layers = full_bert.encoder.layer[model.merge_layer :]

        for i in range(len(original_top_layers)):
            # Instantiate a fresh layer with Cross-Attention enabled
            new_layer = BertLayer(model.head_config)

            # COPY trained weights (Self-Attention + FFN) from original BERT to new layer
            # Note: The Cross-Attention block (new_layer.crossattention) will remain random!
            if not model.random_top_layers:
                logger.info(
                    f"Copying weights from original BERT to Mid-Fusion top layer {i}"
                )
                self._copy_bert_weights(original_top_layers[i], new_layer)
            else:
                logger.info(f"Initializing Mid-Fusion top layer {i} randomly")

            model.top_layers.append(new_layer)

        # pooler
        # Preserve the pretrained pooler to keep the original [CLS] projection
        model.pooler = getattr(full_bert, "pooler", None)

        if model.pooler is None:
            logger.warning(
                "No pooler found in the base model; using [CLS] token directly."
            )

    def _copy_bert_weights(self, src, target):
        """
        Copies Self-Attention and FFN weights from src to target.
        Leaves Cross-Attention weights (only in target) initialized randomly.
        """
        target.attention.self.load_state_dict(src.attention.self.state_dict())
        target.attention.output.load_state_dict(src.attention.output.state_dict())
        target.intermediate.load_state_dict(src.intermediate.state_dict())
        target.output.load_state_dict(src.output.state_dict())
        if hasattr(target, "crossattention") and target.crossattention:
            target.crossattention.self.load_state_dict(src.attention.self.state_dict())
            target.crossattention.output.load_state_dict(
                src.attention.output.state_dict()
            )


class InitMICEModernBERTFromHFID(LightweightTask):
    """Worker-node task to load weights into MICE ModernBERT model"""

    model: Param[ModernBertMiceCrossEncoder]

    def execute(self):
        # Ensure model is instantiated and initialized
        model = self.model
        hf_id = model.hf_id

        # Ensure configs are available
        if not hasattr(model, "config") or model.config is None:
            model.config = AutoConfig.from_pretrained(hf_id)

        if not hasattr(model, "head_config") or model.head_config is None:
            model.head_config = AutoConfig.from_pretrained(hf_id)
            model.head_config.is_decoder = True
            model.head_config.add_cross_attention = True

        # Build the model structure first
        model.initialize()

        logger.info(f"Loading MICE ModernBERT weights from {hf_id}")

        full_backbone = AutoModelForSequenceClassification.from_pretrained(hf_id)
        model.embeddings.load_state_dict(full_backbone.model.embeddings.state_dict())
        for i in range(model.merge_layer):
            model.bottom_layers[i].load_state_dict(
                full_backbone.model.layers[i].state_dict()
            )

        src_layers = full_backbone.model.layers[model.merge_layer :]
        for i, target_layer in enumerate(model.top_layers):
            if not model.random_top_layers:
                self._copy_modernbert_weights(src_layers[i], target_layer)
        model.final_norm.load_state_dict(full_backbone.model.final_norm.state_dict())
        model.head.load_state_dict(full_backbone.head.state_dict())

    def _copy_modernbert_weights(self, src, target):
        """
        Copies Attention and MLP weights from src to target.
        Leaves Cross-Attention weights (only in target) initialized randomly.
        """
        target.attn_norm.load_state_dict(src.attn_norm.state_dict())
        target.attn.load_state_dict(src.attn.state_dict())
        target.mlp_norm.load_state_dict(src.mlp_norm.state_dict())
        target.mlp.load_state_dict(src.mlp.state_dict())
        # Seed cross-attention
        all_head = src.attn.all_head_size
        with torch.no_grad():
            target.crossattention.q_proj.weight.copy_(
                src.attn.Wqkv.weight[0:all_head, :]
            )
            target.crossattention.k_proj.weight.copy_(
                src.attn.Wqkv.weight[all_head : 2 * all_head, :]
            )
            target.crossattention.v_proj.weight.copy_(
                src.attn.Wqkv.weight[2 * all_head : 3 * all_head, :]
            )
            target.crossattention.Wo.load_state_dict(src.attn.Wo.state_dict())


def mice_scorer(
    hf_id: str,
    merge_layer: int = 6,
    drop_layer: int = 0,
    mask_cls_to_doc: bool = True,
    mask_query_to_cls: bool = True,
    freeze_base: bool = False,
    random_top_layers: bool = False,
    compress_dim: float = 1.0,
    pooling_method: Optional[str] = None,
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
    max_length: int = 512,
) -> Tuple[MiceCrossEncoder, List[LightweightTask]]:
    """
    Unified entry point for creating a MICE scorer.
    Automatically selects between BERT and ModernBERT architectures based on the hf_id.

    Args:
        hf_id: Hugging Face checkpoint identifier.
        merge_layer: Layer index where mid-fusion starts.
        drop_layer: Layer index at which to stop (dropping subsequent backbone layers).
        mask_cls_to_doc: If True, prevents [CLS] from attending to document tokens.
        mask_query_to_cls: If True, prevents query tokens from attending to [CLS].
        freeze_base: If True, freezes the bottom layers.
        random_top_layers: If True, initializes top layers randomly.
        compress_dim: Dimensionality compression factor for top layers.
        pooling_method: (ModernBERT only) "cls" or "mean" pooling.
        max_query_length: Maximum number of tokens for the query.
        max_doc_length: Maximum number of tokens for the document.
        max_length: Maximum total number of tokens (used as default for query/doc if not specified).
    """
    tokenizer = MICEQueryDocTokenizer.C(
        model_id=hf_id,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length,
        max_length=max_length,
    )

    if "modernbert" in hf_id.lower() or "ettin" in hf_id.lower():
        model = ModernBertMiceCrossEncoder.C(
            hf_id=hf_id,
            tokenizer=tokenizer,
            merge_layer=merge_layer,
            drop_layer=drop_layer,
            mask_cls_to_doc=mask_cls_to_doc,
            mask_query_to_cls=mask_query_to_cls,
            freeze_base=freeze_base,
            random_top_layers=random_top_layers,
            compress_dim=compress_dim,
            pooling_method=pooling_method,
        )
        return model, [InitMICEModernBERTFromHFID.C(model=model)]
    else:
        model = BertMiceCrossEncoder.C(
            hf_id=hf_id,
            tokenizer=tokenizer,
            merge_layer=merge_layer,
            drop_layer=drop_layer,
            mask_cls_to_doc=mask_cls_to_doc,
            mask_query_to_cls=mask_query_to_cls,
            freeze_base=freeze_base,
            random_top_layers=random_top_layers,
            compress_dim=compress_dim,
        )
        return model, [InitMICEBERTFromHFID.C(model=model)]
