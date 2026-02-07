from enum import Enum
import attrs
from attrs import Factory, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Union,
    Type,
    get_args,
    get_origin,
    get_type_hints,
)
import copy
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration
from functools import cached_property as attrs_cached_property
from xpm_torch.configuration import FabricConfiguration
from itertools import product
import logging
from omegaconf import DictConfig, MISSING

logger = logging.getLogger(__name__)


class Losses(str, Enum):
    """Possible losses"""
    
    BCE = "bce"
    """ Binary Cross Entropy loss """

    hingeLoss = "hingeLoss"
    """Hinge loss"""

    infoNCE = "infoNCE"
    """InfoNCE loss"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    infoNCE_RankDistiLLM = "infoNCE_RankDistiLLM"
    """InfoNCE using the negatives sampled by Schlatt et al. 2025 with ColBERTv2"""

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
        if any(k in d for k in ["value", "values_list", "values_range"]):
            return GenericParams.from_any(d)
    return obj


@configuration()
class GenericParams:
    value: Any = None
    values_list: Optional[List[Any]] = None
    values_range: Optional[Tuple[int, int]] = None

    def as_list(self) -> List[Any]:
        if self.values_list:
            return list(self.values_list)
        if self.values_range:
            return list(range(self.values_range[0], self.values_range[1]))
        if self.value is not None:
            return [self.value]
        return []

    @classmethod
    def from_any(cls, obj: Any, target_type: Type = Any) -> "GenericParams":

        def converter(value: Any) -> Any:
            """Attempts to convert a value to the target_type."""
            if target_type is Any:
                return value

            types_to_try = []
            if get_origin(target_type) is Union:
                types_to_try.extend(get_args(target_type))
            else:
                types_to_try.append(target_type)

            for t in types_to_try:
                if t is type(None):
                    continue
                try:
                    return t(value)
                except (ValueError, TypeError):
                    continue

            return value

        # 1. Already the right type
        if isinstance(obj, cls):
            return obj

        # 2. It's a raw value
        if isinstance(obj, (str, int, float, bool)):
            return cls(value=converter(obj))

        # 3. It's a list
        if isinstance(obj, (list, tuple)):
            return cls(values_list=[converter(v) for v in obj])

        # 4. It's a dict or DictConfig
        if isinstance(obj, (dict, DictConfig)):
            d = dict(obj)
            value = d.get("value")
            values_list = d.get("values_list")

            if value is not None:
                value = converter(value)

            if values_list is not None:
                values_list = [converter(v) for v in values_list]

            return cls(
                value=value,
                values_list=values_list,
                values_range=(
                    tuple(d.get("values_range")) if "values_range" in d else None
                ),
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

    #TODO use FabricConfiguration
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
    base: str = ""
    """Identifier for the base model"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the Ettin based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""

    grid_search: Dict[str, GenericParams] = field(factory=dict)
    """
    Grid search parameters. Maps a dot-separated parameter path to a GenericParams object.
    Example in YAML:
    grid_search:
      learner.optimization.lr:
        values_list: [1e-5, 2e-5]
      pooling_method:
        value: "cls"
    """


def set_nested_attr(obj: Any, path: str, value: Any):
    """Sets a nested attribute on an object."""
    keys = path.split(".")
    current = obj
    for key in keys[:-1]:
        current = getattr(current, key)
    setattr(current, keys[-1], value)


def get_nested_attr_type(obj: Any, path: str) -> Type:
    """
    Traverses a nested object to find the type hint of the final attribute.
    """
    keys = path.split(".")
    current_obj = obj
    for key in keys[:-1]:
        # In case of intermediate GenericParams, we can't traverse further
        if not hasattr(current_obj, key):
            return Any
        current_obj = getattr(current_obj, key)

    try:
        type_hints = get_type_hints(type(current_obj))
        return type_hints.get(keys[-1], Any)
    except Exception:
        return Any


def generate_grid(cfg: Any) -> Tuple[List, List[str]]:
    """
    Generates a list of configuration permutations for a grid search, based
    on a `grid_search` dictionary in the main configuration object.

    The `grid_search` dictionary should map dot-separated attribute paths
    to a list of values or a GenericParams-style object.

    Example `grid_search` in config:
    {
        "learner.optimization.lr": [1e-5, 2e-5],
        "pooling_method": {"value": "cls"}
    }
    returns:
     configs: List[Configs] the list of all configs 
     tagspaths: a unique id per config using fixed params   
    """
    # If grid_search is not present or empty, just return the original config.
    if not hasattr(cfg, "grid_search") or not cfg.grid_search:
        logger.info("no params to grid search, returning raw config")
        return [cfg], [cfg.id]

    grid_params = cfg.grid_search

    param_paths = list(grid_params.keys())

    value_options = []
    for path in param_paths:
        target_type = get_nested_attr_type(cfg, path)

        # This converter is defined locally to have access to the target_type
        def converter(value: Any) -> Any:
            if target_type is Any:
                return value
            types_to_try = [
                t
                for t in (
                    get_args(target_type)
                    if get_origin(target_type) is Union
                    else [target_type]
                )
                if t is not type(None)
            ]
            for t in types_to_try:
                try:
                    return t(value)
                except (ValueError, TypeError):
                    continue
            return value

        # The framework likely pre-instantiates GenericParams, so we get the object,
        # extract its raw values, and convert them just-in-time.
        gp_from_framework = (
            grid_params[path]
            if isinstance(grid_params[path], GenericParams)
            else GenericParams.from_any(grid_params[path])
        )
        raw_values = gp_from_framework.as_list()
        converted_values = [converter(v) for v in raw_values]
        value_options.append(converted_values)

    # Generate Cartesian product of all parameter values
    grid_combinations = product(*value_options)
    
    output_configs = []
    tagspaths = []

    # Create a base configuration to be copied for each permutation
    # and clear its grid_search to make generated configs clean.
    base_cfg = copy.deepcopy(cfg)
    base_cfg.grid_search = {}
    logger.info("Building grid search configs")
    for combination in grid_combinations:
        tagpath = []
        new_cfg = copy.deepcopy(base_cfg)
        for path, value in zip(param_paths, combination):
            tagpath.append(f"{path}={value}")
            set_nested_attr(new_cfg, path, value)
        output_configs.append(new_cfg)
        tagspaths.append("_".join(tagpath))
        logger.info("tagpath: " + tagspaths[-1])

    # If the product is empty (e.g., one of the value lists was empty),
    # this will correctly return an empty list.
    return output_configs, tagspaths
