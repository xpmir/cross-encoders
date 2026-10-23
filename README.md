<div align="center">

<h1>Reproducing and Comparing Distillation Techniques for Cross-Encoders</h1>
<div>
    <a href='https://victormorand.github.io/' target='_blank'>Victor Morand</a><sup>1</sup>&emsp;
    <a href=https://scholar.google.com/citations?user=QGCo1PAAAAAJ&hl target='_blank'>Mathias Vast</a><sup>1,2</sup>&emsp;
    <a target='_blank'>Basile van Cooten</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.fr/citations?user=3gUQp6oAAAAJ&hl' target='_blank'>Laure Soulier</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=V-Nyr0wAAAAJ' target='_blank'>Josiane Mothe</a><sup>3</sup>&emsp;
    <a href='https://www.piwowarski.fr' target='_blank'>Benjamin Piwowarski</a><sup>1</sup>&emsp;
</div>
<br>
<div>
    <sup>1</sup>Sorbonne Université, CNRS, ISIR, F-75005 Paris, France&emsp;<br>
    <sup>2</sup>ChapsVision, Paris, France&emsp;<br>
    <sup>3</sup>IRIT, Université de Toulouse, UMR5505 CNRS, F-31400 Toulouse, France&emsp;<br>
</div>
<br>

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](http://arxiv.org/abs/2603.03010)
[![All Models](https://img.shields.io/badge/🤗%20Hugging%20Face%20Models-blue)](https://huggingface.co/collections/xpmir/reproducing-cross-encoders)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/xpmir/cross-encoders)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Experimaestro](https://img.shields.io/badge/orchestrated%20by-experimaestro-orange)](https://github.com/experimaestro/experimaestro-python)
</div>

### 📄 Paper Abstract

> Recent advances in Information Retrieval have established transformer-based cross-encoders as a keystone in IR. Recent studies have focused on knowledge distillation and showed that, with the right strategy, traditional cross-encoders could reach the level of effectiveness of LLM re-rankers. Yet, comparisons with previous training strategies, including distillation from strong cross-encoder teachers, remain unclear. In addition, few studies cover a similar range of backbone encoders, while substantial improvements have been made in this area since BERT. This lack of comprehensive studies in controlled environments makes it difficult to identify robust design choices. In this work, we reproduce [Schlatt et al. 2025](http://arxiv.org/abs/2405.07920) LLM-based distillation strategy and compare it to [Hofstätter et al. 2020](https://www.semanticscholar.org/paper/Improving-Efficient-Neural-Ranking-Models-with-Hofst%C3%A4tter-Althammer/102f40abbd5f64a0f7c341ceaf8af4fb536e35f8) approach based on an ensemble of cross-encoder teachers, as well as other supervised objectives, to fine-tune a large range of cross-encoders, from the original BERT and its follow-ups RoBERTa, ELECTRA and DeBERTa-v3, to the more recent ModernBERT. We evaluate all models on both in-domain (TREC-DL and MS MARCO dev) and out-of-domain datasets (BEIR, LoTTE, and Robust04). Our results show that objectives emphasizing relative comparisons---pairwise MarginMSE and listwise InfoNCE---consistently outperform pointwise baselines across all backbones and evaluation settings, and that objective choice can yield gains comparable to scaling the backbone architecture.

---

## 👨‍💻 Package Overview

This repository provides a standardized and fully modular framework for fine-tuning, evaluating, and reproducing state-of-the-art Cross-Encoder models for Information Retrieval (IR).
Built upon the [Experimaestro](https://github.com/experimaestro/experimaestro-python) and [xpmir](https://github.com/experimaestro/experimaestro-ir) ecosystems, it automates the end-to-end experimental lifecycle:
- Document indexing and first-stage retrieval (BM25, SPLADE, ColBERT / PLAID).
- Advanced supervised training and knowledge distillation strategies.
- Rigorous, distributed multi-benchmark evaluation (TREC-DL, BEIR, LoTTE, Robust04, NanoBEIR, LIMIT).
- Pretrained models released on Hugging Face: [`xpmir/reproducing-cross-encoders`](https://huggingface.co/collections/xpmir/reproducing-cross-encoders).

---

## 🎯 Implemented Training Strategies

This framework implements a wide spectrum of objective functions and negative sampling schemes across multiple backbone families:

### 1. Distillation Objectives
- **MarginMSE** ([Hofstätter et al. 2020](https://www.semanticscholar.org/paper/Improving-Efficient-Neural-Ranking-Models-with-Hofst%C3%A4tter-Althammer/102f40abbd5f64a0f7c341ceaf8af4fb536e35f8)): Pairwise margin distillation matching teacher score differences $\Delta s = s(q, d^+) - s(q, d^-)$ between positive and negative documents from a cross-encoder teacher ensemble.
- **DistillRankNET** ([Schlatt et al. 2025](http://arxiv.org/abs/2405.07920)): Pairwise distillation targeting soft probability distributions over pair preferences induced by LLM / teacher scores.
- **ADR-MSE (Approximate Distributional Ranking MSE)** ([Schlatt et al. 2025](http://arxiv.org/abs/2405.07920)): Listwise distillation matching the Plackett-Luce distribution over top-$k$ candidates with teacher predictions.
- **Pointwise MSE & Pearson Correlation**: Distillation against dense teacher scores (e.g. `cross-encoder/ettin-reranker-v1-data` with Mixedbread-large, used to train [the ettin reranker family](https://huggingface.co/blog/ettin-reranker)) using pointwise squared error or batch Pearson correlation loss ($1 - r$).

### 2. Supervised & Contrastive Objectives
- **InfoNCE (with In-Batch Negatives)**: Contrastive listwise ranking objective maximizing the likelihood of positive documents against in-batch negatives ([Schlatt et al. 2025](http://arxiv.org/abs/2405.07920)).
- **InfoNCE with Hard Negatives**: InfoNCE formulated over mined hard negatives, including ColBERTv2 hard negatives (`infoNCE_Colbertv2Neg`) and RankDistiLLM top-50 pool negatives (`infoNCE_RankDistiLLM`).
- **Binary Cross-Entropy (BCE)**: Classic pointwise classification loss on positive vs. hard-negative pairs (with optional ColBERTv2 negatives).
- **Hinge Loss**: Margin-based ranking loss enforcing a minimum margin between positive and negative documents.

---

## ⚡ Experimaestro: Caching, Launchers & Parallelism

Experiments in this repository are orchestrated by **[Experimaestro](https://github.com/experimaestro/experimaestro-python)**, which provides unique advantages for compute efficiency and scale:

### 1. First-Stage Result Caching (Save Compute Across Runs)
First-stage candidate generation (e.g., BM25 indexing, SPLADE-v3 top-1000 retrieval, or ColBERT/PLAID indexation) is computationally expensive.
Experimaestro tracks task dependencies using content-addressable identifiers:
- First-stage runs are computed **once** and cached in the workspace.
- All subsequent training experiments, loss comparisons, and evaluations automatically **reuse the exact same cached candidate runs**.
- Switching loss functions, batch sizes, or backbones does not trigger re-indexing or re-retrieval, saving hundreds of GPU and CPU hours.

### 2. Heterogeneous Launchers: Short vs. Long / Multi-GPU
Each stage of the pipeline defines its own resource requirements, enabling fine-grained dispatch to different hardware:
- **Lightweight / Short Tasks** (data prep, validation topic evaluation, DAG inspection): Dispatched to fast queues or single GPUs (`duration=2h & gpu(mem=16G)`).
- **Heavy Training Runs** (large backbones, 80M sample training): Dispatched to multi-GPU nodes or high-memory cluster partitions (`duration=24h & cpu(cores=16) & cuda(mem=80G, gpus=4)`).
- **Indexation & Negative Mining**: Dispatched to high-core CPU or specialized memory tiers.

Experimaestro automatically matches each requirement to your configured local or SLURM environments (via `~/.config/experimaestro/launchers.py`).

### 3. Massive Parallel Benchmark Evaluation
In Experimaestro, every benchmark evaluation is an independent task in the dependency DAG:
When you enable multiple evaluation benchmarks in a configuration:
```yaml
evaluation:
    in_domain: true     # MS MARCO dev, TREC DL 19, TREC DL 20
    beir13: true        # 13 out-of-domain BEIR datasets
    lotte_search: true  # 5 LoTTE search domains
    robust04: true      # Robust04
```
Experimaestro automatically schedules and launches the evaluation of all benchmark datasets **in parallel** across available GPUs/cluster nodes at the very moment the Learner finishes !

---

## 📦 Installation

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic Python environment management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xpmir/cross-encoders.git
   cd cross-encoders
   ```

2. **Synchronize dependencies:**
   For training and cluster execution (runtime dependencies only):
   ```bash
   uv sync
   ```
   To also include development and testing tools (`pytest`, `ruff`, `mypy`):
   ```bash
   uv sync --group dev
   ```

3. **(Optional) Flash-Attention 2** for accelerated attention and unpadded sequence processing is highly recommended for training with longer sequences (e.g ettin-reranker-data with `MSE_mixedbread_large` strategy). You can install it manually or use the HF kernel:
   ```bash
   uvx hf download kernels-community/flash-attn2
   ```

---

## 🔬 Reproducing Paper Experiments

The YAML configurations in `experiments/repro_paper/` cover all model backbones and distillation objectives:

| Configuration | Description |
|---|---|
| [`core_debug.yaml`](experiments/repro_paper/core_debug.yaml) | Fast sanity-check configuration for local debugging |
| [`core_mini_debug.yaml`](experiments/repro_paper/core_mini_debug.yaml) | MiniLM debug configuration with limited query sets |
| [`core_mini.yaml`](experiments/repro_paper/core_mini.yaml) | MiniLM & small backbones (17M, 32M, 68M) across 6 loss functions |
| [`core_base_paper.yaml`](experiments/repro_paper/core_base_paper.yaml) | Base models (BERT, RoBERTa, ELECTRA, DeBERTa-v3, Ettin-150M) |
| [`core_base_deberta.yaml`](experiments/repro_paper/core_base_deberta.yaml) | DeBERTa-v3 base model training |
| [`core_base_ettin.yaml`](experiments/repro_paper/core_base_ettin.yaml) | Ettin base backbone experiments |
| [`core_large.yaml`](experiments/repro_paper/core_large.yaml) | Large backbones and scaling experiments |
| [`reproduction_rank_distiLLM.yaml`](experiments/repro_paper/reproduction_rank_distiLLM.yaml) | Direct reproduction of Rank-DistiLLM training |
| [`Negatives/colbertNeg_base.yaml`](experiments/repro_paper/Negatives/colbertNeg_base.yaml) | Base models trained with ColBERTv2 hard negative mining |
| [`Negatives/RankDistiLLMNeg_base.yaml`](experiments/repro_paper/Negatives/RankDistiLLMNeg_base.yaml) | Base models trained with RankDistiLLM negative pools |

### Validating with a Dry Run
To inspect the tasks and execution DAG without submitting jobs :
```bash
uv run experimaestro run experiments/repro_paper/core_mini_debug.yaml --run-mode DRY_RUN
```

### Launching Experiments
To launch an experiment (local execution or automated SLURM dispatch depending on your launchers):
```bash
uv run experimaestro run experiments/repro_paper/core_base_paper.yaml
```

---

## 📑 Citation

If you find this work or codebase useful, please cite:

```bibtex
@misc{morand2026reproducingcomparingdistillationtechniques,
      title={Reproducing and Comparing Distillation Techniques for Cross-Encoders},
      author={Victor Morand and Mathias Vast and Basile Van Cooten and Laure Soulier and Josiane Mothe and Benjamin Piwowarski},
      year={2026},
      eprint={2603.03010},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2603.03010},
}
```
