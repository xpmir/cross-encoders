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
    using a join_mask.
    """

    join_layer: Param[int] = field(default=6)
    """The layer index at which full self-attention begins."""

    prettr_max_query_length: Param[int] = field(default=32)
    """Legacy PreTTR query length parameter; fixed-offset encoding is kept in garage helper."""

    _version: Constant[int] = 1

    def _garage_shifted_position_ids(
        self, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Garage helper for future fixed-offset PreTTR positional ids."""
        BAT, SEQ = token_type_ids.shape
        position_ids = (
            torch.arange(SEQ, dtype=torch.long, device=token_type_ids.device)
            .unsqueeze(0)
            .expand(BAT, SEQ)
            .clone()
        )

        is_doc = token_type_ids == 1
        for b in range(BAT):
            doc_indices = torch.where(is_doc[b])[0]
            if len(doc_indices) > 0:
                doc_start = doc_indices[0]
                num_doc = len(doc_indices)
                position_ids[b, doc_start:] = torch.arange(
                    self.prettr_max_query_length,
                    self.prettr_max_query_length + num_doc,
                    device=token_type_ids.device,
                )
        return position_ids

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

        BAT, SEQ = input_ids.shape

        # 2. Intercept Backbone Loop
        # We assume standard HF transformer models (BERT, RoBERTa, Electra, etc.)
        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        # Initial embeddings
        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        dtype = hidden_states.dtype

        if (
            hasattr(base_model, "embeddings_project")
            and base_model.embeddings_project is not None
        ):
            hidden_states = base_model.embeddings_project(hidden_states)

        # Build a single attention mask once: padding + join-blocks.
        extended_attention_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        ).to(dtype=dtype)

        # get extended attn mask (2D)
        b_attention_mask = attention_mask.bool()
        ext_attn_mask = b_attention_mask.reshape(
            BAT, 1, SEQ, 1
        ) * b_attention_mask.reshape(BAT, 1, 1, SEQ)
        join_mask = ~ext_attn_mask | (
            token_type_ids.reshape(BAT, 1, SEQ, 1)
            != token_type_ids.reshape(BAT, 1, 1, SEQ)
        )
        join_mask = join_mask.to(dtype=dtype).masked_fill_(
            join_mask, torch.finfo(dtype).min
        )

        for i, layer_module in enumerate(base_model.encoder.layer):
            layer_mask = join_mask if i < self.join_layer else extended_attention_mask
            hidden_states = layer_module(hidden_states, attention_mask=layer_mask)

        # 3. Scoring
        # Match HF model's pooling/classification logic
        # Roberta and Electra classification heads expect the full sequence (3D)
        if model.config.model_type in ["roberta", "electra"]:
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
        # The document part is 'doc [SEP]' (encoded with token_type_id=1).
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

        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
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
            add_special_tokens=False,
            return_tensors="pt",
        )

        sep_id = self.model.tokenizer.tokenizer.sep_token_id
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

        tokenized_obj = TokenizedTexts(
            ids=input_ids,
            mask=attention_mask,
            token_type_ids=torch.ones_like(input_ids),
            lens=attention_mask.sum(dim=1).tolist(),
        )

        return TokensEncoderOutput(
            tokenized_obj,
            self.model.encode_documents(input_ids, attention_mask),
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
