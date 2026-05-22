<div align="center">

<h1>Reproducing and Comparing Distillation Techniques for Cross-Encoders</h1>

<div>
    Anonymous ACL Submission&emsp;<br>
</div>
</div>


### 📄 Paper Abstract

> Recent advances in Information Retrieval have established transformer-based cross-encoders as a keystone in IR. Recent studies have focused on knowledge distillation and showed that, with the right strategy, traditional cross-encoders could reach the level of effectiveness of LLM re-rankers. Yet, comparisons with previous training strategies, including distillation from strong cross-encoder teachers, remain unclear. In addition, few studies cover a similar range of backbone encoders, while substantial improvements have been made in this area since BERT. This lack of comprehensive studies in controlled environments makes it difficult to identify robust design choices. In this work, we reproduce [Schlatt et al. 2025](http://arxiv.org/abs/2405.07920) LLM-based distillation strategy and compare it to [Hofstätter et al. 2020](https://www.semanticscholar.org/paper/Improving-Efficient-Neural-Ranking-Models-with-Hofst%C3%A4tter-Althammer/102f40abbd5f64a0f7c341ceaf8af4fb536e35f8) approach based on an ensemble of cross-encoder teachers, as well as other supervised objectives, to fine-tune a large range of cross-encoders, from the original BERT and its follow-ups RoBERTa, ELECTRA and DeBERTa-v3, to the more recent ModernBERT. We evaluate all models on both in-domain (TREC-DL and MS MARCO dev) and out-of-domain datasets (BEIR, LoTTE, and Robust04). Our results show that objectives emphasizing relative comparisons---pairwise MarginMSE and listwise InfoNCE---consistently outperform pointwise baselines across all backbones and evaluation settings, and that objective choice can yield gains comparable to scaling the backbone architecture.

## 👨‍💻 Package

This repository provides a standardized and fully modular framework for the fine-tuning and evaluation of state-of-the-art Cross-Encoder models for Information Retrieval (IR).
It automates the end-to-end experimental lifecycle—from document indexing (using BM25 or SPLADE) and advanced training strategies (such as MarginMSE and listwise distillation) to rigorous multi-benchmark evaluation on datasets like BEIR.
We release this project as a resource for the IR community, simplifing the reproduction and evaluation of sota re-ranking results, and providing modular components for researching new loss functions and negative sampling techniques.

## What is available for reviewers
While we use an internal and non-anonymous library to conduct experiments, we were forced to anonimize it in the code, making it unusable in this anonymized repository.

It still provides:
- All training configurations avalable as yaml files.
- Paper experiments configurations are gathered by model size in `src/cross_encoders_training/` while the main experimental pipeline is written in `src/cross_encoders_training/experiment.py`
