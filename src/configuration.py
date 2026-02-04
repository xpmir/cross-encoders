from enum import Enum
import attrs
from attrs import Factory, field
from typing import Any, List, Optional, Tuple, TypeVar, Generic, Union, Type, get_args
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration
from functools import cached_property as attrs_cached_property
from xpm_torch.configuration import FabricConfiguration
from itertools import product
import logging
from omegaconf import DictConfig, MISSING


logging.basicConfig(level=logging.INFO)

class Losses(str, Enum):
    """Possible losses"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    BCE = "bce"
    """ Binary Cross Entropy loss """

    distillRankNET = "distillRankNET"
    """Distillation version of RankNET loss from Schlatt et al. 2025"""

    ADR_MSE = "ADR_MSE"
    """Listwise distillation loss proposed by Schlatt et al. 2025"""


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



T = TypeVar("T", int, str, float)

def to_params(obj: Any) -> Any:
    if isinstance(obj, (dict, DictConfig)):
        # If it looks like a GenericParams dict, convert it.
        # A simple heuristic: check for 'value', 'values_list', or 'values_range'.
        d = dict(obj)
        if any(k in d for k in ['value', 'values_list', 'values_range']):
            return GenericParams.from_any(d)
    return obj

@configuration()
class GenericParams:
    value: Any = None
    values_list: Optional[List[Any]] = None
    values_range: Optional[Tuple[int, int]] = None

    def as_list(self) -> List[Any]:
        if self.values_list: return list(self.values_list)
        if self.values_range: return list(range(self.values_range[0], self.values_range[1]))
        if self.value is not None: return [self.value]
        return []
    
    @classmethod
    def from_any(cls, obj: Any) -> "GenericParams":
        # 1. Already the right type
        if isinstance(obj, cls):
            return obj
        
        # 2. It's a raw string/int (base: "model_name")
        if isinstance(obj, (str, int, float)):
            return cls(value=obj)
            
        # 3. It's a list (base: ["model1", "model2"])
        if isinstance(obj, (list, tuple)):
            return cls(values_list=list(obj))
            
        # 4. It's a dict or DictConfig (base: {values_list: [...]})
        if isinstance(obj, (dict, DictConfig)):
            # Convert DictConfig to real dict to avoid OmegaConf attribute errors
            d = dict(obj)
            return cls(
                value=d.get("value"),
                values_list=d.get("values_list"),
                values_range=tuple(d.get("values_range")) if "values_range" in d else None
            )
        
        return cls(value=obj)

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
    base: Any = field(
        default="bert-base-uncased", 
        converter=to_params
    )
    """Identifier for the base model"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the Ettin based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""


def generate_grid(cfg: Any) -> List[Any]:
    """
    Recursively scans a configuration object and returns a list of all 
    possible configuration permutations based on GenericParams fields.
    """
    if isinstance(cfg, GenericParams):
        return cfg.as_list()
    
    if hasattr(cfg, "__attrs_attrs__"):
        fields = attrs.fields(type(cfg))
        field_names = [f.name for f in fields]
        
        field_options = []
        for name in field_names:
            val = getattr(cfg, name)
            field_options.append(generate_grid(val)) # Recurse
            
        grid = []
        for combination in product(*field_options):
            new_instance = type(cfg)(**dict(zip(field_names, combination)))
            grid.append(new_instance)
        return grid

    if isinstance(cfg, list):
        list_options = [generate_grid(item) for item in cfg]
        return [list(res) for res in product(*list_options)]

    return [cfg]