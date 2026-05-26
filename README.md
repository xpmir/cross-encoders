<div align="center">

<h1>MICE: Minimal Interaction Cross-Encoders for efficient Re-ranking</h1>

<div>
    Anonymous ACL Submission&emsp;<br>
</div>
</div>


### 📄 Paper Abstract

>  Cross-encoders deliver state-of-the-art ranking effectiveness but have a high inference cost, limiting their use to second-stage re-rankers.  Prior work has addressed this bottleneck from two largely separate directions: accelerating cross-encoder inference  through attention sparsification, or improving first-stage retrieval effectiveness to alleviate the need of a re-ranker, using more complex models, e.g. late-interactions. In this work, we bridge these two directions through an in-depth analysis of cross-encoder internal mechanisms. By identifying and removing superfluous interactions, we derive MICE (Minimal Interaction Cross-Encoders), a new cross-encoder architecture that retains effectiveness while reducing computational overhead. Extensive evaluations on both in-domain and out-of-domain datasets demonstrate that MICE matches or exceeds its cross-encoder counterpart in effectiveness, while decreasing FLOPs by up to $2.5$ times. 

## What is available for reviewers

This anonymous repo provides:
- All training configurations avalable as yaml files.
- Implementation of Mice models can be found in `src/MICE/modeling`
- Paper experiments configurations are gathered by model size in `src/MICE/experiments/*yaml` while the main experimental pipeline is written in `src/MICE/mice_training.py`

_Because we use an internal and non-anonymous library to conduct experiments, we were forced to anonimize it in the code, making it unusable in this anonymized repository._
