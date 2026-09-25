import tempfile
from pathlib import Path
import torch

from xpmir.test.utils.utils import SampleDocumentStore
from xpmir.index.plaid import PlaidIndex, PlaidIndexBuilder
from xpmir.text.encoders import TextEncoderBase, TokenizedTexts, TokensRepresentationOutput


from experimaestro import Param, field

class MockEncoder(TextEncoderBase):
    """Simple mock encoder for testing PlaidIndexBuilder."""

    dim: Param[int] = field(default=16, ignore_default=True)

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def model(self):
        return None

    def document_token_embeddings(self, documents) -> list[torch.Tensor]:
        outputs = []
        for doc in documents:
            text = getattr(doc, "text", "doc")
            num_tokens = max(len(text.split()), 2)
            outputs.append(torch.randn(num_tokens, self.dim))
        return outputs

    def tokenize(self, texts) -> TokenizedTexts:
        pass

    def encode(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        pass


from experimaestro import ObjectStore
from experimaestro.taskglobals import Env
from experimaestro.xpmutils import DirectoryContext


def test_plaid_index_ext2int_and_batch_tokens(tmp_path: Path):
    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir(exist_ok=True)
    context = DirectoryContext(tmp_path)
    objects = ObjectStore()

    doc_store = SampleDocumentStore.C(num_docs=5)
    encoder = MockEncoder.C(dim=16)

    builder = PlaidIndexBuilder.C(
        documents=doc_store,
        encoder=encoder,
        buffer_size=2,
        n_bits=2,
        kmeans_niters=2,
        compress_only=True,
    )
    builder_instance = builder.instance(context=context, objects=objects)
    builder_instance.execute()

    index = PlaidIndex.C(
        documents=doc_store,
        index_path=builder_instance.index_path,
        compress_only=True,
    ).instance()

    # Test single external string ID (SampleDocumentStore generates IDs "0", "1", "2"...)
    tokens_alpha = index.get_document_tokens("0")
    assert isinstance(tokens_alpha, torch.Tensor)
    assert tokens_alpha.shape[1] == 16

    # Test single integer ID
    tokens_int0 = index.get_document_tokens(0)
    assert isinstance(tokens_int0, torch.Tensor)
    assert tokens_int0.shape == tokens_alpha.shape

    # Test batch list of external string IDs
    batch_tokens = index.get_document_tokens(["0", "2"])
    assert isinstance(batch_tokens, list)
    assert len(batch_tokens) == 2
    assert batch_tokens[0].shape == tokens_alpha.shape

    # Test batch list of integer IDs
    batch_int_tokens = index.get_document_tokens([0, 2])
    assert isinstance(batch_int_tokens, list)
    assert len(batch_int_tokens) == 2
