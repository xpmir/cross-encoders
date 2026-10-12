<div align="center">

<h1>MICE to Meet You: From Re-ranking Only to End-to-End Retrieval</h1>

<div>
    Anonymous ECIR Submission&emsp;<br>
</div>
</div>


### 📄 Paper Abstract

>   Late-interaction models offer state-of-the art efficiency-effectiveness trade-offs for neural retrieval by enabling document representations to be pre-computed and efficiently indexed. However, their relatively simple interaction mechanisms, typically MaxSim, can limit their effectiveness compared to cross-encoders. MICE, a recently introduced re-ranking architecture, aims to combine cross-encoder-like effectiveness through expressive transformer-based scoring with reduced inference-time computation through document-side pre-computation. %However, storing these documents was not used in practice due to the required disk space and poor performance. However, no practical way to store document representations were explored.  We address  this limitation here and propose MICE-P, the first practical MICE retrieval pipeline. It adapts PLAID to index its pre-computed multi-vector document representations. We conduct a controlled comparison of MICE, ColBERT and cross-encoders by training comparable models using the Ettin Family, recently released ModernBERT-based encoders, under matched experimental conditions. Our results show that MICE-P defines a new class of retrieval architecture with an efficiency–effectiveness trade-off of its own: it matches late-interaction retrieval throughput while retaining the effectiveness of cross-encoders.

## What is available for ECIR reviewers

This anonymous repo provides:
- All training configurations avalable as yaml files.
- Implementation of Mice models can be found in `src/MICE/modeling`
- Paper experiments configurations are gathered by model size in `experiments/mice_v2/*yaml` while the main experimental pipeline is written in `src/MICE/experiments/micev2_training.py`

_Because we use an internal and non-anonymous library to conduct experiments, we were forced to anonimize it in the code, making it unusable in this anonymized repository, the full logic as well as all hyperparameters are still fully available and readable._
