import torch
import logging
import tempfile
from pathlib import Path
from MICE.modeling.mice import mice_scorer, MiceCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)


class TestMiceForwardTask(LightweightTask):
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


class TestMiceReloadTask(LightweightTask):
    scorer: Param[MiceCrossEncoder]
    model_id: Param[str]
    merge_layer: Param[int]

    def execute(self):
        """
        Task execution for testing saving and reloading a MICE model.
        """
        print(f"\n--- Testing Model Reload (ID: {id(self.scorer)}) ---")
        self.scorer.initialize()
        self.scorer.eval()

        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        with torch.no_grad():
            original_output = self.scorer(input_records)

        print(f"Original model output (before reload): {original_output}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            print(f"Saving model to {tmp_path}...")
            self.scorer.save_model(tmp_path)

            print("Creating a new model instance and loading weights...")
            # Create a new configuration and instance
            new_scorer_cfg, _ = mice_scorer(
                hf_id=self.model_id, merge_layer=self.merge_layer
            )
            new_scorer = new_scorer_cfg.instance()
            # This NEW instance needs initialization to create the skeleton
            new_scorer.initialize()
            new_scorer.load_model(tmp_path)
            new_scorer.eval()

            with torch.no_grad():
                reloaded_output = new_scorer(input_records)

            print(f"Reloaded model output: {reloaded_output}")

            # Check if outputs are identical
            assert torch.allclose(original_output, reloaded_output), (
                "Reloaded model output differs from original!"
            )

        print("Model reload test successful! Outputs match.")


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

    # Create and run the reload task
    test_reload_task = TestMiceReloadTask.C(
        scorer=scorer_cfg, model_id=model_id, merge_layer=merge_layer
    ).instance()
    # Run initialization tasks if they exist
    for init_task in init_tasks:
        init_task.instance().execute()
    test_reload_task.execute()


if __name__ == "__main__":
    # Use a small BERT model for testing purposes
    # MiniLM is a good candidate as it is fast and small
    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_id = "Qwen/Qwen3-0.6B"
    merge_layer = 3

    test_mice(model_id, merge_layer)
