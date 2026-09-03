import pytest
from unittest.mock import MagicMock
from pathlib import Path

from configuration import CheckpointSelection, Evaluation
from training_utils import find_checkpoint
from xpm_torch.learner import LearnerOutput


class TestCheckpointSelectionNormalization:
    """Tests for Evaluation.get_checkpoint_modes() normalization."""

    def test_single_last(self):
        e = Evaluation(eval_checkpoint="last")
        assert e.get_checkpoint_modes() == [CheckpointSelection.LAST]

    def test_single_negative_one(self):
        e = Evaluation(eval_checkpoint=-1)
        assert e.get_checkpoint_modes() == [CheckpointSelection.LAST]

        e_str = Evaluation(eval_checkpoint="-1")
        assert e_str.get_checkpoint_modes() == [CheckpointSelection.LAST]

    def test_single_val_and_aliases(self):
        for val_alias in ("val", "best", "best_val", "best val"):
            e = Evaluation(eval_checkpoint=val_alias)
            assert e.get_checkpoint_modes() == [CheckpointSelection.VAL]

    def test_single_integer_epoch(self):
        e_int = Evaluation(eval_checkpoint=500)
        assert e_int.get_checkpoint_modes() == [500]

        e_str = Evaluation(eval_checkpoint="500")
        assert e_str.get_checkpoint_modes() == [500]

    def test_list_of_values(self):
        e = Evaluation(eval_checkpoint=["last", "500"])
        assert e.get_checkpoint_modes() == [CheckpointSelection.LAST, 500]

        e2 = Evaluation(eval_checkpoint=["val", 100, "last"])
        assert e2.get_checkpoint_modes() == [
            CheckpointSelection.VAL,
            100,
            CheckpointSelection.LAST,
        ]

    def test_list_deduplication_preserves_order(self):
        e = Evaluation(eval_checkpoint=["last", "val", "last", "val"])
        assert e.get_checkpoint_modes() == [
            CheckpointSelection.LAST,
            CheckpointSelection.VAL,
        ]

    def test_both_raises_value_error(self):
        e = Evaluation(eval_checkpoint="both")
        with pytest.raises(ValueError, match="Option 'both' has been removed"):
            e.get_checkpoint_modes()

        e_list = Evaluation(eval_checkpoint=["val", "both"])
        with pytest.raises(ValueError, match="Option 'both' has been removed"):
            e_list.get_checkpoint_modes()

    def test_invalid_string_raises_error(self):
        e = Evaluation(eval_checkpoint="invalid_checkpoint")
        with pytest.raises(ValueError):
            e.get_checkpoint_modes()


class TestFindCheckpoint:
    """Tests for find_checkpoint() utility."""

    def test_find_in_scheduled_checkpoints(self):
        mock_task_25 = MagicMock(name="task_25")
        mock_task_50 = MagicMock(name="task_50")
        outputs = LearnerOutput(
            listeners={},
            learned_model=MagicMock(name="learned_model"),
            checkpoints={25: mock_task_25, 50: mock_task_50},
        )

        # Int lookup
        result = find_checkpoint(outputs, 25)
        assert result == mock_task_25

        # Str lookup
        result_str = find_checkpoint(outputs, "50")
        assert result_str == mock_task_50

    def test_match_max_epochs_fallback(self):
        mock_learned_model = MagicMock(name="learned_model")
        outputs = LearnerOutput(
            listeners={},
            learned_model=mock_learned_model,
            checkpoints={25: MagicMock(), 50: MagicMock()},
        )
        mock_learner = MagicMock(name="learner")
        mock_learner.max_epochs = 100

        result = find_checkpoint(outputs, 100, learner=mock_learner)
        assert result == mock_learned_model

    def test_disk_fallback_when_not_in_scheduled(self, tmp_path):
        from xpm_torch.trainers.context import TrainerContext, TrainState

        # Setup mock disk structure: checkpoint-00000075/model
        chk_dir = TrainerContext.get_checkpoint_path(tmp_path, 75)
        model_dir = chk_dir / TrainState.MODEL_DIR
        model_dir.mkdir(parents=True)

        mock_loader_task = MagicMock(name="loader_task_75")
        mock_learner = MagicMock(name="learner")
        mock_learner.checkpointspath = tmp_path
        mock_learner.max_epochs = 200
        mock_learner.checkpoint_interval = 25
        mock_learner.model.loader_config.return_value = mock_loader_task

        outputs = LearnerOutput(
            listeners={},
            learned_model=MagicMock(),
            checkpoints={25: MagicMock(), 50: MagicMock()},
        )

        result = find_checkpoint(outputs, 75, learner=mock_learner)
        assert result == mock_loader_task
        mock_learner.model.loader_config.assert_called_once()

    def test_not_found_raises_informative_error(self):
        outputs = LearnerOutput(
            listeners={},
            learned_model=MagicMock(),
            checkpoints={25: MagicMock(), 50: MagicMock()},
        )
        mock_learner = MagicMock(name="learner")
        mock_learner.checkpointspath = Path("/nonexistent/path")
        mock_learner.checkpoint_interval = 25
        mock_learner.max_epochs = 100

        with pytest.raises(ValueError) as exc_info:
            find_checkpoint(outputs, 73, learner=mock_learner)

        error_msg = str(exc_info.value)
        assert "73" in error_msg
        assert "[25, 50]" in error_msg
        assert "checkpoint_interval is 25" in error_msg
