#!/usr/bin/env python3
"""
Verification Script for DDP Multi-GPU Training Infrastructure.

Verifies:
1. Multi-GPU strategy check in build_trainer (auto/ddp -> ddp_find_unused_parameters_true).
2. Compute-agnostic Trainer initialization.
3. TrainerContext runtime DDP batch size division and logging.
4. DDP multi-process gradient synchronization and model parameter parity.
"""

import sys
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from configuration import CE_FineTuning
from training_utils import build_trainer
from xpm_torch.base import Sampler
from xpm_torch.trainers.context import TrainerContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ddp_training")


class DummyDataset(Dataset):
    def __init__(self, size=64):
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        # Add an unused parameter to verify find_unused_parameters=True
        self.unused_param = nn.Parameter(torch.randn(5))

    def forward(self, x):
        return self.linear(x)


class DummySampler(Sampler):
    pass


def test_build_trainer_warning():
    logger.info("=== Testing build_trainer Multi-GPU Strategy Resolution ===")
    cfg = CE_FineTuning()
    cfg.learner.fabric.devices = 2
    cfg.learner.fabric.strategy = "auto"
    cfg.learner.loss = "marginMSE"
    cfg.learner.optimization.batch_size = 64

    # Patch sampler creation with a dummy Sampler Config object
    with patch("training_utils.msmarco_hofstaetter_ensemble_hard_negatives") as mock_sampler:
        mock_sampler.return_value = DummySampler.C()
        trainer = build_trainer(cfg)

    assert (
        cfg.learner.fabric.strategy == "ddp_find_unused_parameters_true"
    ), f"Expected strategy 'ddp_find_unused_parameters_true', got {cfg.learner.fabric.strategy}"
    assert trainer.batch_size == 64, f"Expected Trainer batch_size to stay 64 (global), got {trainer.batch_size}"
    logger.info("PASSED: Strategy auto-updated to 'ddp_find_unused_parameters_true' and Trainer remains compute-agnostic with global batch_size=64!\n")


def test_trainer_context_local_bs():
    logger.info("=== Testing TrainerContext get_local_batch_size ===")
    mock_fabric = MagicMock()
    mock_fabric.world_size = 4
    mock_fabric.is_global_zero = True

    ctx = TrainerContext(
        logpath=Path("/tmp"),
        path=Path("/tmp"),
        max_epoch=1,
        steps_per_epoch=1,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=mock_fabric,
    )

    local_bs = ctx.get_local_batch_size(128)
    assert local_bs == 32, f"Expected local batch size 32, got {local_bs}"
    logger.info(f"PASSED: TrainerContext calculated local_bs={local_bs} from global_bs=128 across 4 devices!\n")


def ddp_worker(fabric: Fabric, global_bs: int):
    world_size = fabric.world_size
    rank = fabric.global_rank

    dataset = DummyDataset(size=128)
    local_bs = max(1, global_bs // world_size)

    if fabric.is_global_zero:
        logger.info(
            f"[DDP Worker] DDP active (world size = {world_size}): "
            f"setting local batch size to {local_bs} (global batch size: {global_bs})"
        )

    assert local_bs == 32, f"Rank {rank}: expected local_bs 32, got {local_bs}"

    dataloader = DataLoader(dataset, batch_size=local_bs, shuffle=False)
    dataloader = fabric.setup_dataloaders(dataloader)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = fabric.setup(model, optimizer)

    criterion = nn.MSELoss()

    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        fabric.backward(loss)
        optimizer.step()
        if batch_idx == 0:
            break

    # Gather linear weights from all ranks to ensure parameters remain in sync
    weights = model.module.linear.weight.data if hasattr(model, "module") else model.linear.weight.data
    gathered_weights = fabric.all_gather(weights)

    if fabric.is_global_zero:
        diff = torch.abs(gathered_weights[0] - gathered_weights[1]).max().item()
        assert diff < 1e-5, f"Weight mismatch across DDP ranks: max diff = {diff}"
        logger.info(f"[DDP Worker] Rank parameter parity verified! Max diff across ranks = {diff:.6f}")


def test_ddp_execution():
    logger.info("=== Testing Multi-Process DDP Execution with Unused Parameters ===")
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_find_unused_parameters_true")
    fabric.launch(ddp_worker, global_bs=64)
    logger.info("PASSED: Multi-process DDP execution verified successfully!\n")


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
        logger.info("Starting DDP Training Infrastructure Verification...")
        test_build_trainer_warning()
        test_trainer_context_local_bs()
        test_ddp_execution()
        logger.info("ALL DDP VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
