# Cross-Encoders

## MonoBERT

Reproduces **Passage Re-ranking with BERT (Rodrigo Nogueira, Kyunghyun Cho). 2019. https://arxiv.org/abs/1901.04085**.

- `monobert/normal`: training with bert-base-uncased. Hugging-face: [xpmir/monobert](https://huggingface.co/xpmir/monobert)

## MonoT5

Reproduces **Nogueira, R., Jiang, Z., Lin, J., 2020. Document Ranking with a Pretrained Sequence-to-Sequence Model.** https://arxiv.org/abs/2003.06713

- `monot5/normal`: training with t5-base, using `true` and `false` tokens for the answers.
