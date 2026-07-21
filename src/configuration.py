from enum import Enum
import copy
import attr
from attrs import Factory, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Type,
    get_args,
    get_origin,
    get_type_hints,
    Annotated,
)
from pydantic import BeforeValidator
from xpm_torch.experiments.configuration import TransformerOptimization, Fabric
from xpmir.papers import configuration
from xpmir.experiments.helpers import LauncherSpecification
from xpmir.datasets.msmarco import RerankerMSMarcoV1Configuration
from itertools import product
import logging

logger = logging.getLogger(__name__)


class Losses(str, Enum):
    """Possible losses"""

    BCE = "bce"
    """ Binary Cross Entropy loss """

    hingeLoss = "hingeLoss"
    """Hinge loss"""

    infoNCE = "infoNCE"
    """InfoNCE loss, with in-batch negatives"""

    infoNCE_Colbertv2Neg = "infoNCE_Colbertv2Neg"
    """InfoNCE using the negatives sampled by Schlatt et al. 2025 with ColBERTv2"""

    infoNCE_RankDistiLLM = "infoNCE_RankDistiLLM"
    """InfoNCE using the negatives sampled from the RankDistiLLM top-50 pool using MS MARCO qrels"""

    BCE_Colbertv2Neg = "BCE_Colbertv2Neg"
    """Binary Cross Entropy loss with ColBERTv2 negatives"""

    hingeLoss_Colbertv2Neg = "hingeLoss_Colbertv2Neg"
    """Hinge loss with ColBERTv2 negatives"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    distillRankNET = "distillRankNET"
    """Distillation version of RankNET loss from Schlatt et al. 2025"""

    ADR_MSE = "ADR_MSE"
    """Listwise distillation loss proposed by Schlatt et al. 2025"""

    MSE_mixedbread_large = "MSE_mixedbread_large"
    """Plain pointwise MSE loss on cross-encoder/ettin-reranker-v1-data teacher scores"""


class PoolingMethod(str, Enum):
    """Possible pooling methods"""

    CLS = "cls"
    """CLS token pooling"""

    MEAN = "mean"
    """Mean pooling"""


class Validation(str, Enum):
    """Possible validation subsets"""

    MSMARCO = "msmarco"
    """Nano MSMARCO validation subset"""

    NanoBEIR = "nanobeir"
    """A small subset of BEIR datasets designed specifically for validation"""

    NanoBEIR11 = "nanobeir11"
    """NanoBEIR excluding argana and touche-2020 as done in [Sentence-Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/evaluation.html#crossencodernanobeirevaluator)"""

    ALL = "all"
    """ Both Nano MSMARCO and NanoBEIR validations"""


T = TypeVar("T", int, str, float)


def to_params(obj: Any) -> Any:
    if isinstance(obj, dict):
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

    @property
    def is_grid(self) -> bool:
        """Returns True if this defines a search space (multiple values)"""
        return len(self.as_list()) > 1

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

        # 4. It's a dict
        if isinstance(obj, dict):
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

        return cls(value=converter(obj))


def _coerce_to_generic_params(v: Any) -> Any:
    if isinstance(v, GenericParams):
        return v
    return GenericParams.from_any(v)


GridSearch = Annotated[
    Union[T, GenericParams],
    BeforeValidator(_coerce_to_generic_params),
]
"""
Type alias for configuration fields supporting inline grid search.

This type allows a configuration field to accept either a single scalar value
or a grid search definition (e.g., list of values) directly in the YAML.
These values are automatically coerced into a `GenericParams` object during
Pydantic validation, and later unpacked into scalar values during grid generation.

Example Usage in a Configuration Class:
    @configuration()
    class MyConfig(Config):
        # Default scalar value
        learning_rate: GridSearch[float] = 1e-5
        batch_size: GridSearch[int] = 32

Example Usage in YAML:
    # 1. Scalar value (no grid search, regular run)
    learning_rate: 1e-5

    # 2. Inline grid search (will generate 2 separate experiment configurations)
    learning_rate: [1e-5, 2e-5]

    # 3. Range-based grid search (using GenericParams dict structure)
    batch_size: {"values_range": [16, 64]}
"""


@configuration()
class Indexation(LauncherSpecification):
    batch_size: int = 512
    max_indexed: int = 0

    requirements: str = "duration=2 days & cpu(cores=8)"
    sparse2bmp_requirements: str = "duration=1d & cuda(mem=24G)"


@configuration()
class Retrieval:
    k: GridSearch[int] = 1000
    batch_size: GridSearch[int] = 128
    requirements: str = "duration=2 days & cuda(mem=24G)"


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=4)"


@configuration()
class Evaluation:
    """What datasets to evaluate on, eventually limit the number of queries for debug"""

    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""

    all_datasets: bool = False
    """Whether to evaluate on all BEIR datasets (minus the 5 not publicly available)"""

    in_domain: bool = False
    """Whether to evaluate on in-domain datasets (MSMarco, TREC DL 19 and 20)"""

    beir13: bool = False
    """Whether to evaluate on all BEIR13 datasets"""

    lotte_search: bool = False
    """Whether to evaluate on all LOTTE Search datasets"""

    robust04: bool = False
    """Whether to evaluate on Robust04"""

    nanobeir: bool = False
    """Whether to evaluate on NanoBEIR datasets"""

    datasets: List[str] = Factory(list)
    """List of specific datasets to evaluate on"""


@configuration()
class xpm_torch_Learner:
    validation_interval: GridSearch[int] = field(default=32)

    validation_top_k: GridSearch[int] = 1000

    checkpoint_interval: GridSearch[int] = field(default=32)

    optimization: TransformerOptimization = Factory(TransformerOptimization)

    requirements: str = "duration=4 days & cuda(mem=24G) * 2"

    sample_rate: GridSearch[float] = 1.0
    """Sample rate for triplets"""

    sample_max: GridSearch[int] = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    max_grad_norm: GridSearch[float] = 0.0
    """Maximum gradient norm (0 for no clipping)"""

    loss: GridSearch[str] = Losses.marginMSE.value
    """Loss function to use"""

    validation: GridSearch[str] = Validation.NanoBEIR.value
    """ The validation subset to use """

    early_stop_epochs: GridSearch[int] = 0
    """ number of **epochs** without improvements before early stopping based on validation"""

    fabric: Fabric = Factory(Fabric)
    """Configuration for Fabric device management"""


@configuration()
class PlaidConfiguration:
    """
    Configuration for PLAID integration in MICE retrieval pipeline.
    """

    use_plaid: bool = False
    """Whether to use PLAID for retrieval"""

    ### Indexation params ###
    buffer_size: int = 1000
    """Number of documents to use for creating/updating the PLAID index"""

    batch_size: int = 25_000
    """Batch size (in tokens) when encoding documents for PLAID"""

    dim: int = 128
    """Per-token embedding dimension for PLAID index"""

    n_bits: int = 2
    """Number of bits for residual quantization in PLAID"""

    kmeans_niters: int = 4
    """Number of K-means iterations for PLAID clustering"""

    n_samples_kmeans: int = 0
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    max_points_per_centroid: int = 256
    """Maximum number of points (documents) per centroid. Controls the creation of new centroids."""

    compress_only: bool = False
    """Whether to build a compress-only index (no IVF search)"""

    force_cpu_indexing: bool = False
    """When True, forces the use of CPU for indexing even if a GPU is available.
    This can be useful to avoid GPU OOM errors during indexing, especially for large corpora."""

    ### Retrieval params ###
    n_ivf_probe: int = 8
    """Number of IVF clusters to probe in PLAID (lower = faster, less accurate)"""

    n_full_scores: int = 0
    """Number of candidates for which fast-plaid computes full scores
    (0 = fast-plaid default)."""


@configuration()
class CE_FineTuning(RerankerMSMarcoV1Configuration):
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""

    attn_implementation: Optional[str] = field(default=None)
    """Attention implementation for HuggingFace models (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    ettin_subset_exclude: List[str] = field(factory=list)
    """The subset config names of cross_encoder.ettin_reranker_v1_data to exclude from training"""

    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    learner: xpm_torch_Learner = Factory(xpm_torch_Learner)

    preprocessing: Preprocessing = Factory(Preprocessing)

    evaluation: Evaluation = Factory(Evaluation)

    plaid: PlaidConfiguration = Factory(PlaidConfiguration)

    ## Retriever Model
    retriever: str = ""
    """Identifier for the retriever model. If empty, uses BM25."""

    precompute_first_stage: bool = True
    """If true, will save the run for the first stage - just reload it for evaluation rather than recomputing"""

    ## Cross Encoder Model
    base: GridSearch[str] = ""
    """Identifier for the base model"""

    max_length: Optional[int] = None
    """max len for scorer, default to 0 = max len of the model"""

    max_query_length: Optional[int] = None
    """Maximum query length for cross-encoders"""

    max_doc_length: Optional[int] = None
    """Maximum document length for cross-encoders"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the ModernBert based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""

    save_runs: bool = False
    """Whether to save the evaluation runs in the best model folders"""

    export_trained_models: bool = True
    """Whether to export the best models to the models/ folder"""

    normalize_docs_per_batch: bool = True
    """whether to normalize documents per batch for listwise losses"""

    grid_search: Dict[str, GridSearch[Any]] = field(factory=dict)
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


def get_nested_attr_type(obj: Any, path: str) -> Any:
    """
    Traverses a nested object to find the type hint of the final attribute.
    Raises ValueError if the path is invalid.
    """
    keys = path.split(".")
    current_obj = obj
    for i, key in enumerate(keys[:-1]):
        if not hasattr(current_obj, key):
            raise ValueError(
                f"Invalid grid search path '{path}': "
                f"'{key}' not found in {type(current_obj).__name__} "
                f"(at level {'.'.join(keys[:i]) if i > 0 else 'root'})"
            )
        current_obj = getattr(current_obj, key)

    last_key = keys[-1]
    if not hasattr(current_obj, last_key):
        raise ValueError(
            f"Invalid grid search path '{path}': "
            f"'{last_key}' not found in {type(current_obj).__name__} "
            f"(at level {'.'.join(keys[:-1]) if len(keys) > 1 else 'root'})"
        )

    try:
        type_hints = get_type_hints(type(current_obj))
        return type_hints.get(last_key, Any)
    except Exception:
        return Any


def discover_grid_params(obj: Any, prefix: str = "") -> Dict[str, GenericParams]:
    """Recursively find all GenericParams instances in an attrs object"""
    found = {}
    if attr.has(type(obj)):
        for f in attr.fields(type(obj)):
            val = getattr(obj, f.name)
            path = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(val, GenericParams) and val.is_grid:
                found[path] = val
            elif val is not None and not isinstance(val, (str, int, float, bool, Enum)):
                found.update(discover_grid_params(val, path))
    return found


def finalize_config(obj: Any):
    """Recursively convert all non-grid GenericParams to scalars"""
    if attr.has(type(obj)):
        for f in attr.fields(type(obj)):
            val = getattr(obj, f.name)
            if isinstance(val, GenericParams) and not val.is_grid:
                setattr(obj, f.name, val.value)
            elif val is not None and not isinstance(val, (str, int, float, bool, Enum)):
                finalize_config(val)


def generate_grid(cfg: Any) -> Tuple[List, List[dict]]:
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
     tags: a list of dicts with the same length as configs, where each dict contains the parameter values that were set for that config. For example:
     [
        {"learner.optimization.lr": 1e-5, "pooling_method": "cls"},
        {"learner.optimization.lr": 2e-5, "pooling_method": "cls"}
     ]
    """
    # 1. Discover inline grid parameters
    grid_params = discover_grid_params(cfg)

    # 2. Merge with explicit grid_search block
    if hasattr(cfg, "grid_search") and cfg.grid_search:
        for path, gp in cfg.grid_search.items():
            if isinstance(gp, GenericParams):
                grid_params[path] = gp
            else:
                grid_params[path] = GenericParams.from_any(gp)

    # If no grid parameters found, just return the original config.
    if not grid_params:
        logger.info("no params to grid search, returning raw config")
        new_cfg = copy.deepcopy(cfg)
        finalize_config(new_cfg)
        return [new_cfg], [{}]

    param_paths = list(grid_params.keys())

    value_options = []
    for path in param_paths:
        # get target type for this parameter from the config class using the path
        # raises an error if the attribute is invalid
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
    tags = []
    # Create a base configuration to be copied for each permutation
    # and clear its grid_search to make generated configs clean.
    base_cfg = copy.deepcopy(cfg)
    base_cfg.grid_search = {}
    logger.info("Building grid search configs")
    for combination in grid_combinations:
        cfg_tags = {}
        new_cfg = copy.deepcopy(base_cfg)
        for path, value in zip(param_paths, combination):
            cfg_tags[path] = value
            set_nested_attr(new_cfg, path, value)

        # Convert any remaining single-value GenericParams to scalars
        finalize_config(new_cfg)

        output_configs.append(new_cfg)
        tags.append(cfg_tags)

    # If the product is empty (e.g., one of the value lists was empty),
    # this will correctly return an empty list.
    return output_configs, tags
