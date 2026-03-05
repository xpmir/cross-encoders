import logging
from pathlib import Path
import torch
import torch.nn as nn
from experimaestro import (
    Config,
    Param,
    Meta,
    Constant,
    DataPath,
    serialize,
    deserialize,
)
from xpm_torch.utils.huggingface import xpmTorchHubModule


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)


    # Example child module
    class MyTorchxpmTorchHubModule(
        xpmTorchHubModule,
        library_name="my-org/my-model",
        tags=["torch", "experimaestro"],
        repo_url="https://github.com/VictorMorand/test-model",
        paper_url="https://arxiv.org/abs/???",
    ):
        """Example xpmTorchHubModule implementation, now all saving and loading is taken care of under the hood"""

        ## Child parameters
        input_dim: Param[int] = 100
        """Input dimension"""

        hidden_dim: Param[int] = 200
        """Hidden dimension"""

        output_dim: Param[int] = 10
        """Output dimension"""

        version: Constant[str] = "1.0"

        def __post_init__(self):
            super().__post_init__()
            # self._config = self.__config__

            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
            logging.debug("Initialized layers: %s, %s", self.fc1, self.fc2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    model_id = "VictorMorand/test-model"
    save_path = Path("./test_model")

    create_new_model = False
    create_new_model = True

    if create_new_model:
        # create configuration
        cfg = MyTorchxpmTorchHubModule.C(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule.from_kwargs(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule(input_dim=50, hidden_dim=100, output_dim=5)

        model: MyTorchxpmTorchHubModule = cfg.instance()
        print(model)
        xpm_config = model.__config__
        print(type(xpm_config),xpm_config)
        
        # test it
        input = torch.randn(1, 50)
        output = model(input)

        print(output)

        model.save_pretrained(save_path)
        # model.push_to_hub(model_id)

        # try to reload it
        logging.info("Reloading model from %s", save_path)
        newModel = MyTorchxpmTorchHubModule.from_pretrained(save_path)

    else:
        # save_path = Path("./test_model")
        # try to load from Hub

        newModel = MyTorchxpmTorchHubModule.from_pretrained(
            model_id, force_download=True, local_files_only=False
        )

    logging.info("New model: %s", newModel)
    logging.info("New model config: %s", newModel.__config__)

    # # # Model that can be re-used in experiments
    # # model, init_tasks = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE")

    # # init_tasks[0].execute()
    # # Use this if you want to actually use the model
    # model = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE", as_instance=True)
    # print(model.rsv("walgreens store sales average", "The average Walgreens salary ranges..."))
