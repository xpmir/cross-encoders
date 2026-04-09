import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
import logging
from MICE.modeling.mice import mice_scorer, MiceCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param


class TestMiceForwardTask(LightweightTask):
    """Tests Mice Loading, scoring a PointWiseItem, and compare outpus after reloading for checkpointing"""

    scorer: Param[MiceCrossEncoder]

    def execute(self):
        """
        Task execution for instantiating a dataset and running a forward pass for a MICE model.
        """
        print(f"Step 1: Instantiating the MICE model (ID: {id(self.scorer)})...")

        # Initialize the model components (experimaestro way)
        # Safe to call multiple times now because it's idempotent in BertMiceCrossEncoder
        self.scorer.initialize()

        print("Step 2: Preparing the dataset...")
        # We use PointwiseItems.from_texts to easily create a small dataset
        # This matches the query-document pair format expected by the model
        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]

        # Create the input records
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        print("Step 3: Running a forward pass...")
        # Switch to evaluation mode
        self.scorer.eval()

        # Run forward pass without gradient computation
        with torch.no_grad():
            # The model's forward method accepts BaseItems (like PointwiseItems)
            output = self.scorer(input_records)

        print(f"Model output (score): {output}")

        # Validation
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"

        print(
            "\nForward pass successful! MICE model is correctly instantiated and working."
        )


def test_mice(model_id, merge_layer):
    """tests mice loading, forward pass saving."""
    # mice_scorer returns the model and a list of initialization tasks
    # that need to be executed to load the weights
    scorer_cfg, init_tasks = mice_scorer(hf_id=model_id, merge_layer=merge_layer)

    print(scorer_cfg)

    # Create and run the forward pass task
    test_forward_task = TestMiceForwardTask.C(scorer=scorer_cfg).instance()
    # Run initialization tasks if they exist
    for init_task in init_tasks:
        init_task.instance().execute()
    test_forward_task.execute()


if __name__ == "__main__":
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)

    # MiniLM is a good candidate as it is fast and small
    model_ids = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "Qwen/Qwen3-0.6B"]
    merge_layer = 3

    for model_id in model_ids:
        print("\n\n ### TESTING ", model_id)
        test_mice(model_id, merge_layer)
