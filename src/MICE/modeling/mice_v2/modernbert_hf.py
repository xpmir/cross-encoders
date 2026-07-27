"""
ModernBERT MICE HuggingFace Implementation.
Custom PreTrainedModel and Modules supporting SDPA, Flash Attention 2, and ST export.
"""

import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

from transformers.models.modernbert.modeling_modernbert import (
    ModernBertModel,
    ModernBertMLP,
    ModernBertAttention,
    ModernBertPreTrainedModel,
    eager_attention_forward,
)
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import is_flash_attn_2_available

try:
    from transformers.modeling_flash_attention_utils import _unpad_input, _pad_input
except ImportError:
    _unpad_input, _pad_input = None, None

logger = logging.getLogger(__name__)


def is_fa2_available() -> bool:
    """Checks if FlashAttention 2 is available via either `flash_attn` package or `kernels` library."""
    if is_flash_attn_2_available():
        return True
    try:
        import kernels  # noqa - HuggingFace kernels library providing kernels-community/flash-attn2

        return True
    except ImportError:
        pass
    try:
        import flash_attn  # noqa

        return True
    except ImportError:
        pass
    return False


class ModernBertMiceConfig(ModernBertConfig):
    model_type = "modernbert_mice"

    def __init__(
        self,
        n_contextualization_layers: int = 6,
        n_docs_ctx_layers: Optional[int] = None,
        n_interaction_layers: Optional[int] = None,
        bound_bottom_layers: bool = True,
        cross_attn_first: bool = True,
        num_labels: int = 1,
        **kwargs,
    ):
        if num_labels == 1:
            kwargs.pop("id2label", None)
            kwargs.pop("label2id", None)
            kwargs["id2label"] = {0: "LABEL_0"}
            kwargs["label2id"] = {"LABEL_0": 0}
        super().__init__(num_labels=num_labels, **kwargs)
        self.n_contextualization_layers = n_contextualization_layers
        self.n_docs_ctx_layers = n_docs_ctx_layers
        self.n_interaction_layers = n_interaction_layers
        self.bound_bottom_layers = bound_bottom_layers
        self.cross_attn_first = cross_attn_first


class ModernBertMiceCrossAttention(nn.Module):
    """
    Cross-attention module adapting ModernBERT's fused Wqkv linear projection
    to perform Cross-Attention (Query attending to Document).
    Supports SDPA and Flash Attention 2 backends via HF's ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.deterministic_flash_attn = getattr(
            config, "deterministic_flash_attn", False
        )

        # Fused Wqkv projection matching ModernBERT
        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * self.head_dim * config.num_attention_heads,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        q_hidden: torch.Tensor,
        d_hidden: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, q_len, _ = q_hidden.shape
        _, d_len, _ = d_hidden.shape

        # Extract Q from q_hidden
        q_qkv = self.Wqkv(q_hidden).view(
            batch_size, q_len, 3, self.num_heads, self.head_dim
        )
        query_states = q_qkv[:, :, 0].transpose(1, 2)  # [B, heads, q_len, head_dim]

        # Extract K, V from d_hidden
        d_qkv = self.Wqkv(d_hidden).view(
            batch_size, d_len, 3, self.num_heads, self.head_dim
        )
        key_states = d_qkv[:, :, 1].transpose(1, 2)  # [B, heads, d_len, head_dim]
        value_states = d_qkv[:, :, 2].transpose(1, 2)  # [B, heads, d_len, head_dim]

        if q_mask is None:
            q_mask = torch.ones(
                (batch_size, q_len), dtype=torch.bool, device=q_hidden.device
            )
        if d_mask is None:
            d_mask = torch.ones(
                (batch_size, d_len), dtype=torch.bool, device=d_hidden.device
            )

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        is_flash_impl = "flash" in str(attn_impl).lower()

        if (
            is_flash_impl
            and q_hidden.is_cuda
            and _unpad_input is not None
            and _pad_input is not None
        ):
            # FlashAttention varlen path (supports flash_attention_2 and kernels-community/flash-attn2)
            q_unpadded, indices_q, cu_seqlens_q, max_seqlen_q, _ = _unpad_input(
                q_hidden, q_mask
            )
            d_unpadded, _, cu_seqlens_d, max_seqlen_d, _ = _unpad_input(
                d_hidden, d_mask
            )

            # Re-project unpadded Q, K, V
            q_un_qkv = self.Wqkv(q_unpadded).view(-1, 3, self.num_heads, self.head_dim)
            Q_un = q_un_qkv[:, 0].unsqueeze(0)  # [1, total_q, heads, head_dim]

            d_un_qkv = self.Wqkv(d_unpadded).view(-1, 3, self.num_heads, self.head_dim)
            K_un = d_un_qkv[:, 1].unsqueeze(0)  # [1, total_d, heads, head_dim]
            V_un = d_un_qkv[:, 2].unsqueeze(0)  # [1, total_d, heads, head_dim]

            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_impl, eager_attention_forward
            )

            attn_output, _ = attn_interface(
                self,
                Q_un.transpose(1, 2),  # [1, heads, total_q, head_dim]
                K_un.transpose(1, 2),  # [1, heads, total_d, head_dim]
                V_un.transpose(1, 2),  # [1, heads, total_d, head_dim]
                attention_mask=None,  # Crucial: bypasses internal 2D _upad_input inside HF/kernels!
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.head_dim**-0.5,
                is_causal=False,
                cu_seq_lens_q=cu_seqlens_q.to(torch.int32),
                cu_seq_lens_k=cu_seqlens_d.to(torch.int32),
                max_length_q=int(max_seqlen_q),
                max_length_k=int(max_seqlen_d),
                **kwargs,
            )

            # Reshape back to [1, total_q, hidden_size] and pad back
            attn_output_flat = attn_output.reshape(-1, self.config.hidden_size)
            attn_output = _pad_input(attn_output_flat, indices_q, batch_size, q_len)
        else:
            if is_flash_impl:
                logger.warning(
                    f"[MICE CrossAttention] FlashAttention requested ('{attn_impl}'), but unpadding is unavailable "
                    f"(_unpad_input={_unpad_input is not None}). Falling back to SDPA with 4D cross-mask."
                )

            # SDPA / Eager path with 4D boolean cross-attention mask
            cross_mask = (
                q_mask.unsqueeze(-1).bool() & d_mask.unsqueeze(1).bool()
            ).unsqueeze(1)

            # 4D cross-attention masks are ONLY supported by SDPA or Eager (NOT any FlashAttention variant).
            # Force 'sdpa' (or 'eager') so FlashAttention interfaces are never called with 4D cross_mask tensors.
            fallback_impl = "eager" if attn_impl == "eager" else "sdpa"
            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                fallback_impl, eager_attention_forward
            )
            attn_output, _ = attn_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=cross_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.head_dim**-0.5,
                is_causal=False,
                **kwargs,
            )
            attn_output = attn_output.reshape(batch_size, q_len, -1).contiguous()

        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output


class ModernBertMiceTopLayer(nn.Module):
    """
    Top interaction layer for MICE: Cross-Attention + Self-Attention + MLP.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.cross_attn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.cross_attn = ModernBertMiceCrossAttention(config, layer_idx=layer_idx)

        self.self_attn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.self_attn = ModernBertAttention(config, layer_idx=layer_idx)

        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        q_hidden: torch.Tensor,
        d_hidden: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
        q_attn_mask: Optional[torch.Tensor] = None,
        q_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_last_layer: bool = False,
    ) -> torch.Tensor:
        # 1. Cross Attention
        cross_out = self.cross_attn(
            self.cross_attn_norm(q_hidden),
            d_hidden,
            q_mask=q_mask,
            d_mask=d_mask,
        )
        q_hidden = q_hidden + cross_out

        # 2. Self Attention
        self_out, _ = self.self_attn(
            self.self_attn_norm(q_hidden),
            position_embeddings=q_position_embeddings,
            attention_mask=q_attn_mask,
        )
        q_hidden = q_hidden + self_out

        # 3. MLP (Slice to CLS token on final top layer to optimize compute)
        if is_last_layer:
            q_hidden = q_hidden[:, 0:1, :]

        q_hidden = q_hidden + self.mlp(self.mlp_norm(q_hidden))

        return q_hidden


class ModernBertMiceForSequenceClassification(ModernBertPreTrainedModel):
    """
    HF-compatible Sequence Classification Model for MICE with ModernBERT backbone.
    Supports standard ST CrossEncoder forward pass as well as independent Q/D pass.
    """

    config_class = ModernBertMiceConfig

    def __init__(self, config: ModernBertMiceConfig):
        attn_impl = getattr(config, "_attn_implementation", "sdpa")
        if "flash" in str(attn_impl).lower():
            fa2_avail = is_fa2_available()
            if not fa2_avail or not torch.cuda.is_available():
                logger.warning(
                    f"Flash Attention 2 requested ('{attn_impl}'), but is not supported in this environment "
                    f"(CUDA available: {torch.cuda.is_available()}, FA2 kernels available: {fa2_avail}). "
                    f"Falling back to `_attn_implementation='sdpa'`."
                )
                config._attn_implementation = "sdpa"

        super().__init__(config)
        self.config = config

        # Standard ModernBERT backbone for bottom layers & embeddings
        self.model = ModernBertModel(config)

        # Prune bottom layers to n_contextualization_layers so bottom layers run only n_context layers
        n_context = config.n_contextualization_layers
        if n_context is not None and n_context < len(self.model.layers):
            logger.info(
                f"Pruning ModernBertModel bottom layers from {len(self.model.layers)} "
                f"to n_contextualization_layers={n_context}"
            )
            self.model.layers = self.model.layers[:n_context]

        # Top interaction layers
        num_top = config.n_interaction_layers
        if num_top is None:
            num_top = config.num_hidden_layers - config.n_contextualization_layers

        self.top_layers = nn.ModuleList(
            [ModernBertMiceTopLayer(config, layer_idx=i) for i in range(num_top)]
        )

        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels if hasattr(config, "num_labels") else 1,
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_ids: Optional[torch.LongTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "return_dict", True)
        )

        # Handle ST concatenated inputs if query_ids is not passed
        if query_ids is None and input_ids is not None:
            sep_token_id = getattr(
                self.config, "sep_token_id", getattr(self.config, "eos_token_id", 50283)
            )

            # Extract query and doc sequences by finding first SEP token per row
            b_size, seq_len = input_ids.shape
            q_ids_list, d_ids_list = [], []
            q_mask_list, d_mask_list = [], []

            for i in range(b_size):
                row = input_ids[i]
                sep_idx = (row == sep_token_id).nonzero(as_tuple=False)
                if len(sep_idx) > 0:
                    split = sep_idx[0].item() + 1
                    q_ids_list.append(row[:split])
                    d_ids_list.append(row[split:])
                    if attention_mask is not None:
                        q_mask_list.append(attention_mask[i, :split])
                        d_mask_list.append(attention_mask[i, split:])
                else:
                    half = seq_len // 2
                    q_ids_list.append(row[:half])
                    d_ids_list.append(row[half:])
                    if attention_mask is not None:
                        q_mask_list.append(attention_mask[i, :half])
                        d_mask_list.append(attention_mask[i, half:])

            # Pad sub-batches
            query_ids = torch.nn.utils.rnn.pad_sequence(
                q_ids_list, batch_first=True, padding_value=self.config.pad_token_id
            )
            doc_ids = torch.nn.utils.rnn.pad_sequence(
                d_ids_list, batch_first=True, padding_value=self.config.pad_token_id
            )
            if attention_mask is not None:
                query_mask = torch.nn.utils.rnn.pad_sequence(
                    q_mask_list, batch_first=True, padding_value=0
                )
                doc_mask = torch.nn.utils.rnn.pad_sequence(
                    d_mask_list, batch_first=True, padding_value=0
                )

        # 1. Encode queries and docs through bottom layers
        q_outputs = self.model(input_ids=query_ids, attention_mask=query_mask, **kwargs)
        d_outputs = self.model(input_ids=doc_ids, attention_mask=doc_mask, **kwargs)

        q_hidden = q_outputs.last_hidden_state
        d_hidden = d_outputs.last_hidden_state

        # Compute position embeddings for query self-attention
        position_ids = torch.arange(
            q_hidden.shape[1], device=q_hidden.device
        ).unsqueeze(0)
        q_pos_emb = self.model.rotary_emb(
            q_hidden, position_ids, layer_type="full_attention"
        )

        # Prepare 4D bidirectional mask for self-attention

        q_attn_mask = None
        if query_mask is not None:
            q_attn_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=q_hidden,
                attention_mask=query_mask,
            )

        # 2. Pass query hidden states through top layers with cross-attention to doc
        num_top = len(self.top_layers)
        for i, top_layer in enumerate(self.top_layers):
            q_hidden = top_layer(
                q_hidden,
                d_hidden,
                q_mask=query_mask,
                d_mask=doc_mask,
                q_attn_mask=q_attn_mask,
                q_position_embeddings=q_pos_emb,
                is_last_layer=(i == num_top - 1),
            )

        # 3. Pooling (CLS token score)
        cls_hidden = q_hidden[:, 0]
        logits = self.classifier(cls_hidden)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register("modernbert_mice", ModernBertMiceConfig)
AutoModel.register(ModernBertMiceConfig, ModernBertMiceForSequenceClassification)
AutoModelForSequenceClassification.register(
    ModernBertMiceConfig, ModernBertMiceForSequenceClassification
)
