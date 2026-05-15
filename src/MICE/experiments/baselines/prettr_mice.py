import torch
import logging
from typing import List, Optional, Tuple

from experimaestro import Param, field, Constant
from datamaestro_ir.data.base import IDTextRecord
from xpmir.letor.records import BaseItems
from xpmir.text import TokenizedTexts
from xpmir.text.tokenizers import TokenizerOptions
from xpmir.text.encoders import (
    EncoderOutput,
    TextEncoderBase,
    TokensEncoderOutput,
    TokensRepresentationOutput,
)
from xpmir.neural.huggingface import HFCrossScorer, InitCEFromHFID
from xpm_torch.utils import to_device

logger = logging.getLogger(__name__)


class PreTTRCrossEncoder(HFCrossScorer):
    """
    PreTTR Baseline implementation wrapped in the MICE framework.
    It uses joint tokenization but prevents cross-attention in the early layers
    using a join_mask. It also supports MICE's encode_documents API via
    fixed-offset positional embeddings.
    """

    join_layer: Param[int] = field(default=6)
    """The layer index at which full self-attention begins."""

    prettr_max_query_length: Param[int] = field(default=32)
    """The fixed offset used for offline document precomputation."""

    _version: Constant[int] = 1

    def forward(
        self,
        inputs: BaseItems,
        tokenized: Optional[TokenizedTexts] = None,
    ):
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        input_ids = to_device(tokenized.ids, self.device)
        attention_mask = to_device(tokenized.mask, self.device)
        token_type_ids = to_device(tokenized.token_type_ids, self.device)

        # 1. Calculate position_ids with fixed offset for document
        BAT, SEQ = input_ids.shape
        position_ids = (
            torch.arange(SEQ, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(BAT, SEQ)
            .clone()
        )

        is_doc = token_type_ids == 1
        if is_doc.any():
            for b in range(BAT):
                doc_indices = torch.where(is_doc[b])[0]
                if len(doc_indices) > 0:
                    doc_start = doc_indices[0]
                    num_doc = len(doc_indices)
                    position_ids[b, doc_start:] = torch.arange(
                        self.prettr_max_query_length,
                        self.prettr_max_query_length + num_doc,
                        device=self.device,
                    )

        # 2. Intercept Backbone Loop
        # We assume standard HF transformer models (BERT, RoBERTa, Electra, etc.)
        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        # Construct standard HF extended attention mask
        # (Handles padding tokens)
        extended_attention_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        # Initial embeddings
        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # Construct join_mask (Prevents cross-type attention)
        # Different token types (0 vs 1) should not attend to each other
        join_mask = token_type_ids.reshape(BAT, 1, SEQ, 1) != token_type_ids.reshape(
            BAT, 1, 1, SEQ
        )
        # Mask is added to scores, so use large negative value
        # Use a value compatible with the model's dtype
        join_mask = join_mask.to(dtype=hidden_states.dtype) * -10000.0

        # Iterate layers manually
        for i, layer_module in enumerate(base_model.encoder.layer):
            if i < self.join_layer:
                # Add join_mask to prevent cross-attention in early layers
                layer_mask = extended_attention_mask + join_mask
            else:
                # Standard full self-attention in late layers
                layer_mask = extended_attention_mask

            layer_outputs = layer_module(hidden_states, attention_mask=layer_mask)
            hidden_states = layer_outputs[0]

            # Ensure hidden_states stays 3D (batch, seq, dim)
            # Some HF versions might squeeze batch if it's 1 in certain paths
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        # 3. Scoring
        # Match HF model's pooling/classification logic
        if model.config.model_type == "roberta":
            # RobertaClassificationHead extracts CLS internally
            logits = model.classifier(hidden_states)
        else:
            # Standard BERT-like: use pooler if exists, otherwise CLS token
            if hasattr(base_model, "pooler") and base_model.pooler is not None:
                pooled_output = base_model.pooler(hidden_states)
            else:
                pooled_output = hidden_states[:, 0]

            if hasattr(model, "dropout"):
                pooled_output = model.dropout(pooled_output)

            logits = model.classifier(pooled_output)

        return logits.squeeze(-1)

    # --- MICE API Compatibility ---

    def query_token_embeddings(self, records: List[IDTextRecord]) -> List[torch.Tensor]:
        # Queries are encoded normally (type 0, pos 0...)
        options = TokenizerOptions(max_length=self.prettr_max_query_length)
        texts = [r.text if hasattr(r, "text") else r["text"] for r in records]
        tokenized = self.tokenizer.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=options.max_length,
            return_tensors="pt",
        )
        output = self.encode_queries(
            tokenized["input_ids"], tokenized["attention_mask"]
        )
        return [output[i] for i in range(output.shape[0])]

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        # For PreTTR, document encoding must match the document part of joint tokenization
        # Standard joint tokenization for BERT is [CLS] query [SEP] doc [SEP]
        # The document part is 'doc [SEP]' (encoded with token_type_id=1 and fixed position offset)
        max_len = getattr(self.tokenizer, "max_doc_length", 512)

        # We encode WITHOUT [CLS] to match the 'type=1' behavior in joint forward
        texts = [r.text if hasattr(r, "text") else r["text"] for r in records]
        tokenized = self.tokenizer.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Manually add [SEP] at the end to match joint tokenization [CLS] query [SEP] doc [SEP]
        sep_id = self.tokenizer.tokenizer.sep_token_id
        input_ids = torch.cat(
            [
                tokenized["input_ids"],
                torch.full(
                    (tokenized["input_ids"].shape[0], 1), sep_id, dtype=torch.long
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                tokenized["attention_mask"],
                torch.ones((tokenized["attention_mask"].shape[0], 1), dtype=torch.long),
            ],
            dim=1,
        )

        output = self.encode_documents(input_ids, attention_mask)
        return [output[i] for i in range(output.shape[0])]

    def encode_queries(self, input_ids, attention_mask):
        input_ids = to_device(input_ids, self.device)
        attention_mask = to_device(attention_mask, self.device)
        token_type_ids = torch.zeros_like(input_ids)

        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        hidden_states = base_model.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        ext_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        for i in range(self.join_layer):
            hidden_states = base_model.encoder.layer[i](
                hidden_states, attention_mask=ext_mask
            )[0]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    def encode_documents(self, input_ids, attention_mask):
        input_ids = to_device(input_ids, self.device)
        attention_mask = to_device(attention_mask, self.device)
        token_type_ids = torch.ones_like(input_ids)

        # Position IDs starting from offset
        seq_length = input_ids.size(1)
        position_ids = (
            torch.arange(
                self.prettr_max_query_length,
                self.prettr_max_query_length + seq_length,
                dtype=torch.long,
                device=self.device,
            )
            .unsqueeze(0)
            .expand_as(input_ids)
        )

        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        ext_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        for i in range(self.join_layer):
            hidden_states = base_model.encoder.layer[i](
                hidden_states, attention_mask=ext_mask
            )[0]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    def get_document_encoder(self) -> TextEncoderBase:
        return PreTTRDocumentEncoder.C(model=self)


class PreTTRDocumentEncoder(TextEncoderBase):
    """Document encoder using PreTTR independent layers"""

    model: Param[PreTTRCrossEncoder]

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def dimension(self):
        return self.model.encoder.model.config.hidden_size

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        # Replicate MiceDocumentEncoder logic
        max_len = getattr(self.model.tokenizer, "max_doc_length", 512)
        tokenized = self.model.tokenizer.tokenizer(
            [r.text for r in records],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # Convert to TokenizedTexts
        tokenized_obj = TokenizedTexts(
            ids=tokenized["input_ids"],
            mask=tokenized["attention_mask"],
            token_type_ids=tokenized.get("token_type_ids"),
            lens=tokenized["attention_mask"].sum(dim=1).tolist(),
        )

        return TokensEncoderOutput(
            tokenized_obj,
            self.model.encode_documents(
                tokenized["input_ids"], tokenized["attention_mask"]
            ),
        )

    def forward(
        self,
        inputs: List[IDTextRecord],
        *args,
        options: Optional[TokenizerOptions] = None,
    ) -> EncoderOutput:
        return self.encode_documents(inputs)


def prettr_scorer(
    hf_id: str,
    join_layer: int,
    max_length: int = 512,
    prettr_max_query_length: int = 32,
) -> Tuple[PreTTRCrossEncoder, List[InitCEFromHFID]]:
    """Factory function for PreTTR scorer"""
    from xpmir.neural.huggingface import (
        HFSequenceClassification,
        HFConfigID,
        HFQueryDocTokenizer,
    )

    encoder = HFSequenceClassification.C(config=HFConfigID.C(hf_id=hf_id))
    tokenizer = HFQueryDocTokenizer.C(model_id=hf_id, max_length=max_length)

    scorer = PreTTRCrossEncoder.C(
        encoder=encoder,
        tokenizer=tokenizer,
        join_layer=join_layer,
        prettr_max_query_length=prettr_max_query_length,
    )

    return scorer, [InitCEFromHFID.C(model=encoder)]
