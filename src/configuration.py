from enum import Enum
from attrs import Factory, field
from typing import Any, List, Optional, Tuple
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration
from functools import cached_property as attrs_cached_property

import logging

logging.basicConfig(level=logging.INFO)

class Losses(str, Enum):
    """Possible losses"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    PointWiseMSE = "PointWiseMSE"
    """ Point Wise Mean Squared Error loss """

class PoolingMethod(str, Enum):
    """Possible pooling methods"""

    CLS = "cls"
    """CLS token pooling"""

    MEAN = "mean"
    """Mean pooling"""

class Validation(str, Enum):
    """Possible validation subsets"""

    MSMARCO = "msmarco"
    """MSMARCO dev set"""

    NanoBEIR = "nanobeir"
    """A small subset of BEIR datasets designed specifically for validation"""

@configuration()
class Indexation(LauncherSpecification):
    batch_size: int = 512
    max_indexed: int = 0

    requirements: str = "duration=2 days & cpu(cores=8)"
    sparse2bmp_requirements: str = "duration=1d & cuda(mem=24G)"

@configuration()
class xpm_torch_Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    checkpoint_interval: int = field(default=32)

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    sample_rate: float = 1.0
    """Sample rate for triplets"""

    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    max_grad_norm: float = 0.0
    """Maximum gradient norm (0 for no clipping)"""

    loss: str = Losses.marginMSE.value
    """Loss function to use"""

    validation: str = Validation.MSMARCO.value
    """ The validation subset to use """

    ## Lighnting Fabric parameters see https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.fabric.Fabric.html#lightning.fabric.fabric.Fabric 

    strategy: str = "auto"
    """Distributed training strategy"""

    precision: Optional[str] = None
    """Precision to use - e.g., '16-mixed', 'bf16-mixed', etc."""

    accelerator: str = "auto"
    """ Accelerator to use """


@configuration()
class Retrieval:
    k: int = 1000
    batch_size: int = 128
    requirements: str = "duration=2 days & cuda(mem=24G)"


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=4)"

@configuration()
class Evaluation:
    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""

    in_domain_only: bool = False
    """Whether to evaluate only on in-domain datasets (MSMarco, TREC DL 19 and 20)"""
    
    all_datasets: bool = False
    """Whether to evaluate on all BEIR datasets (minus the 5 not publicly available)"""

@configuration()
class CE_FineTuning(RerankerMSMarcoV1Configuration):
    
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""
    
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    learner: xpm_torch_Learner = Factory(xpm_torch_Learner)
    
    preprocessing: Preprocessing = Factory(Preprocessing)

    evaluation: Evaluation = Factory(Evaluation)

    ## Retriever Model
    retriever: str = ""
    """Identifier for the retriever model. If empty, uses BM25."""
    
    ## Cross Encoder Model
    base: str = "bert-base-uncased"
    """Identifier for the base model"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the Ettin based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""