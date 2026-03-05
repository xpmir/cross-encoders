from experimaestro.experiments import mock_modules
import logging

logger = logging.getLogger(__name__)

# Modules to mock (submodules are automatically included)
# this avoid the heavy importing of some lib for the main experiment task that only launches the real jobs
modules_to_mock = [
            "torch",
            "torchmetrics",
            "torchdata",
            "pytorch_lightning",
            "lightning",
            "sentence_transformers",
            "transformers",
            "huggingface_hub.hub_mixin",
            "pylate",
            "xpmir.learning.losses",
            "xpm_torch.datasets",
        ]
logger.info(f"Mocking (not importing) the following modules in experiment manager: {modules_to_mock}")

mock_modules( modules_to_mock,
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