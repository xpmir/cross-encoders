import torch
import logging
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
        print("Step 1: Instantiating the MICE model...")

        # Initialize the model components (experimaestro way)
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


if __name__ == "__main__":
    # Use a small BERT model for testing purposes
    # MiniLM is a good candidate as it is fast and small
    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # mice_scorer returns the model and a list of initialization tasks
    # that need to be executed to load the weights
    scorer_cfg, init_tasks = mice_scorer(hf_id=model_id, merge_layer=3)

    # Run initialization tasks if they exist
    for init_task in init_tasks:
        init_task.instance().execute()

    print(scorer_cfg)

    # Create and run the task
    test_task = TestMiceForwardTask.C(scorer=scorer_cfg).instance()
    test_task.execute()
