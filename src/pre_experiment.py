"""
Experiment Optimization Module for Orchestration.

This module uses experimaestro's mocking capabilities to prevent heavy libraries
(like torch, transformers, and pytorch_lightning) from being imported by the
main experiment manager process.

By mocking these dependencies, the manager can rapidly plan, schedule, and
launch worker jobs without the overhead of loading deep learning frameworks.
The actual heavy dependencies are only imported within the worker processes
where they are required for execution.
"""

from experimaestro.experiments import mock_modules
import logging
import experimaestro.experiments.mockmodule as mm

logger = logging.getLogger(__name__)

# Ensure _FakeClassProxy has __bases__ = () for issubclass compatibility
# (e.g. in datasets/utils/_dill.py: issubclass(obj_type, torch.Tensor))
_orig_proxy_init = mm._FakeClassProxy.__init__


def _proxy_init_with_bases(self, fake_class):
    _orig_proxy_init(self, fake_class)
    object.__setattr__(self, "__bases__", ())


mm._FakeClassProxy.__init__ = _proxy_init_with_bases

# Modules to mock (submodules are automatically included)
# this avoid the heavy importing of some lib for the main experiment task that only launches the real jobs
modules_to_mock = [
    "torch",
    "torchmetrics",
    "torchdata",
    "lightning",
    "pytorch_lightning",
    "lightning_fabric",
    "sentence_transformers",
    "transformers",
    "safetensors.torch",
    "pylate",
    "xpmir.learning.losses",
    "xpmir.neural._sparton",
    "xpm_torch.datasets",
    "MICE.modeling.mice_v2.modernbert_hf",
]
logger.info(
    f"Mocking (not importing) the following modules in experiment manager: {modules_to_mock}"
)

mock_modules(
    modules_to_mock,
    # Decorators to make no-ops
    decorators=[
        "torch.compile",
        "torch.jit.script",
        "torch.jit.unused",
        "torch.jit.export",
        "torch.jit.ignore",
        "torch.no_grad",
        "torch.inference_mode",
    ],
)

import torch  # noqa: E402

torch.__version__ = "2.5.0"
