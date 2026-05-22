<div align="center">

<h1>Building Better Encoder-only Cross-Encoders: A Controlled Study of Training Strategies for Neural Re-ranking</h1>

<div>
    Anonymous ACM Submission&emsp;<br>
</div>
</div>


### 📄 Paper Abstract

>  Cross-encoder re-rankers fine-tuned from Transformer backbones remain the standard for second-stage retrieval, and recent knowledge-distillation recipes have closed much of the gap with LLM re-rankers. Yet the relative merits of these recipes—distillation from LLM rankers versus from strong cross-encoder teachers, and against purely supervised objectives—have not been compared under controlled conditions, and the contribution of newer backbones (RoBERTa, ELECTRA, DeBERTaV3, ModernBERT) versus the original BERT is unclear. We run 162 controlled training runs (9 backbones x 6 objectives x 3 seeds), spanning pointwise, pairwise, and listwise losses with both human labels and two distillation signals, and evaluate on TREC-DL, MS~MARCO dev, BEIR, and LoTTE. We find that objectives emphasizing relative comparisons—pairwise MarginMSE and listwise InfoNCE distillation— consistently outperform pointwise baselines across all backbones, and switching objective yields gains comparable to moving up one backbone size tier.

## 👨‍💻 Package

This repository provides a standardized and fully modular framework for the fine-tuning and evaluation of state-of-the-art Cross-Encoder models for Information Retrieval (IR).
It automates the end-to-end experimental lifecycle—from document indexing (using BM25 or SPLADE) and advanced training strategies (such as MarginMSE and listwise distillation) to rigorous multi-benchmark evaluation on datasets like BEIR.
We release this project as a resource for the IR community, simplifing the reproduction and evaluation of sota re-ranking results, and providing modular components for researching new loss functions and negative sampling techniques.

## What is available for reviewers

This anonymous repo provides:
- All training configurations avalable as yaml files.
- Paper experiments configurations are gathered by model size in `src/cross_encoders_training/` while the main experimental pipeline is written in `src/cross_encoders_training/experiment.py`
- The full experiments results processing Notebook `CrossEncoderResults.py` with the figures generated for the preprint.

_Because we use an internal and non-anonymous library to conduct experiments, we were forced to anonimize it in the code, making it unusable in this anonymized repository._
