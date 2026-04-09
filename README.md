<div align="center">

<h1>Reproducing and Comparing Distillation Techniques for Cross-Encoders</h1>
<div>
    <a href='https://victormorand.github.io/' target='_blank'>Victor Morand</a><sup>1</sup>&emsp;
    <a href=https://scholar.google.com/citations?user=QGCo1PAAAAAJ&hl target='_blank'>Mathias Vast</a><sup>12</sup>&emsp;
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
</div>

### 📄 Paper Abstract 

> Recent advances in Information Retrieval have established transformer-based cross-encoders as a keystone in IR. Recent studies have focused on knowledge distillation and showed that, with the right strategy, traditional cross-encoders could reach the level of effectiveness of LLM re-rankers. Yet, comparisons with previous training strategies, including distillation from strong cross-encoder teachers, remain unclear. In addition, few studies cover a similar range of backbone encoders, while substantial improvements have been made in this area since BERT. This lack of comprehensive studies in controlled environments makes it difficult to identify robust design choices. In this work, we reproduce [Schlatt et al. 2025](http://arxiv.org/abs/2405.07920) LLM-based distillation strategy and compare it to [Hofstätter et al. 2020](https://www.semanticscholar.org/paper/Improving-Efficient-Neural-Ranking-Models-with-Hofst%C3%A4tter-Althammer/102f40abbd5f64a0f7c341ceaf8af4fb536e35f8) approach based on an ensemble of cross-encoder teachers, as well as other supervised objectives, to fine-tune a large range of cross-encoders, from the original BERT and its follow-ups RoBERTa, ELECTRA and DeBERTa-v3, to the more recent ModernBERT. We evaluate all models on both in-domain (TREC-DL and MS MARCO dev) and out-of-domain datasets (BEIR, LoTTE, and Robust04). Our results show that objectives emphasizing relative comparisons---pairwise MarginMSE and listwise InfoNCE---consistently outperform pointwise baselines across all backbones and evaluation settings, and that objective choice can yield gains comparable to scaling the backbone architecture.

## 👨‍💻 Package 

This repository provides a standardized and fully modular framework for the fine-tuning and evaluation of state-of-the-art Cross-Encoder models for Information Retrieval (IR). 
Built upon the Experimaestro and xpmir ecosystems, it automates the end-to-end experimental lifecycle—from document indexing (using BM25 or SPLADE) and advanced training strategies (such as MarginMSE and listwise distillation) to rigorous multi-benchmark evaluation on datasets like BEIR. 
We release this project as a resource for the IR community, simplifing the reproduction and evaluation of sota re-ranking results, and providing modular components for researching new loss functions and negative sampling techniques.

## Reproducing Paper results

The use of [experimaestro](https://github.com/experimaestro/experimaestro-python) ( and especially its IR extension [experimaestro-ir](https://github.com/experimaestro/experimaestro-ir)) ensures full reproducibility, all results from the paper can be reproduced in one command.

- All training configurations are avalable as yaml files.
- Paper experiments are gathered by model size in `src/ir_training/paper`
- We also provide a demo configuration for reproducing only [`🤗xpmir/cross-encoder-ettin-150m-infoNCE`](https://huggingface.co/xpmir/cross-encoder-ettin-150m-infoNCE): `ettin150_training.yaml`

- We release all trained models in the HF collection [`xpmir/reproducing-cross-encoders`](https://huggingface.co/collections/xpmir/reproducing-cross-encoders) [![All Models](https://img.shields.io/badge/🤗%20Hugging%20Face%20Models-blue)](https://huggingface.co/collections/xpmir/reproducing-cross-encoders)


# Usage 

## Installation

To install this repository, first ensure you have `git` and [`uv`](https://docs.astral.sh/uv/) installed.

1.  Clone the repository and its submodules:
    ```bash
    git clone --recurse-submodules git@git.isir.upmc.fr:morand/sota-cross-encoders.git
    cd sota-cross-encoders
    ```
    If you have already cloned the repository without `--recurse-submodules`, you can initialize and update them with:
    ```bash
    git submodule update --init --recursive
    ```

2.  Synchronize the Python dependencies using [`uv`](https://docs.astral.sh/uv/) - this will create a virtual environment with the exact requirements in 
    ```bash
    uv sync
    ```


### Launching experiments
If it is the first time using experimaestro, you may first setup your environment to be able to launch experiments. You can follow [the tutorial](https://experimaestro-python.readthedocs.io/en/latest/tutorial.html) 


Once experimaestro is setup, you can launch the training of Ettin-150m cross-encoder with the following command:

```
uv run experimaestro run-experiment src/ir_training/ettin150_training.yaml
```

You may fist want to see what tasks will be launched with:
```
uv run experimaestro run-experiment src/ir_training/ettin150_training.yaml --run-mode DRY_RUN
```


### Citation 


If you find this work useful, you can cite our work as:


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
