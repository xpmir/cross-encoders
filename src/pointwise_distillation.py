"""Pointwise distillation sampler and trainer for Experimaestro-IR compatibility."""

from typing import List, TypedDict, Iterator
from typing_extensions import ReadOnly
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from experimaestro import Param, field, Config
from xpm_torch.trainers import LossTrainer, TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.datasets import ShardedIterableDataset, InfiniteDataset
from xpm_torch.base import Sampler
from datamaestro_ir.data.distillation import (
    PointwiseDistillationSample,
    PointwiseDistillationSamples,
)
from xpmir.letor.records import PointwiseItems, PointwiseItem


class PointwiseDistillationInputs(TypedDict):
    records: ReadOnly[PointwiseItems]
    teacher_scores: ReadOnly[torch.Tensor]
    tokenized_records: ReadOnly[dict]


def pointwise_distillation_collate(
    samples: List[PointwiseDistillationSample],
) -> PointwiseDistillationInputs:
    records = PointwiseItems()
    teacher_scores = []
    for sample in samples:
        records.add(
            PointwiseItem(sample.query, sample.document.document, sample.document.score)
        )
        teacher_scores.append(sample.document.score)
    return PointwiseDistillationInputs(
        records=records,
        teacher_scores=torch.tensor(teacher_scores, dtype=torch.float32),
        tokenized_records=None,
    )


class PreShuffledPointwiseDataset(ShardedIterableDataset):
    def __init__(self, combined_hf, sample_builder):
        super().__init__()
        self.hf_data = combined_hf
        self.sample_builder = sample_builder

    def iter_shard(
        self, shard_id: int, num_shards: int
    ) -> Iterator[PointwiseDistillationSample]:
        # Use contiguous=True to guarantee sequential disk reads for the worker
        shard = self.hf_data.shard(
            num_shards=num_shards, index=shard_id, contiguous=True
        )
        for row in shard:
            yield self.sample_builder._build_sample(row)


class PointwiseDistillationSampler(Sampler):
    """Sampler wrapper for pointwise distillation datasets"""

    samples: Param[list[PointwiseDistillationSamples]]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)
        self.rng = random

    def as_dataset(self) -> ShardedIterableDataset:
        from datasets import concatenate_datasets

        hf_datasets = []
        for s in self.samples:
            if not hasattr(s, "data"):
                raise ValueError(
                    "All samples must be HuggingFace datasets with a .data property for pre-shuffling."
                )
            hf_datasets.append(s.data)

        # Concatenate, shuffle (with fixed seed across ranks to avoid fingerprint mismatch), and flatten to disk cache
        seed = self.rng.randint(0, 2**32 - 1)
        combined_hf = (
            concatenate_datasets(hf_datasets).shuffle(seed=seed).flatten_indices()
        )

        # Pass the first sample config as the builder (assumes identical schemas)
        return InfiniteDataset(
            PreShuffledPointwiseDataset(combined_hf, self.samples[0])
        )


class DistillationPointwiseLoss(Config, nn.Module):
    """The abstract loss for pointwise distillation"""

    weight: Param[float] = field(default=1.0, ignore_default=True)
    NAME = "?"

    def initialize(self, ranker):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"pointwise-{self.NAME}", loss, self.weight))

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

            student_scores: A batch tensor of student predictions
            teacher_scores: A batch tensor of teacher scores
        """
        raise NotImplementedError()


class PointwiseMSELoss(DistillationPointwiseLoss):
    """Plain pointwise MSE loss between student and teacher scores"""

    NAME = "MSE"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> Tensor:
        return self.loss(student_scores, teacher_scores)


class PointwiseDistillationTrainer(LossTrainer):
    """Trainer for pointwise distillation"""

    lossfn: Param[DistillationPointwiseLoss]
    """The distillation pointwise batch function"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationPointwiseLoss):
            loss.initialize(self.model)

        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset()

        if hasattr(self.model, "get_tokenizer_fn"):
            tokenization_fn = self.model.get_tokenizer_fn()

            def collate_fn_with_tokenization(
                samples: List[PointwiseDistillationSample],
            ) -> PointwiseDistillationInputs:
                inputs = pointwise_distillation_collate(samples)
                inputs["tokenized_records"] = tokenization_fn(inputs["records"])
                return inputs

            collate_fn = collate_fn_with_tokenization
        else:
            collate_fn = pointwise_distillation_collate

        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: PointwiseDistillationInputs):
        records = inputs["records"]
        teacher_scores = inputs["teacher_scores"]
        tokenized_records = inputs.get("tokenized_records")

        if tokenized_records is not None:
            student_scores = self.model(records, tokenized=tokenized_records)
        else:
            student_scores = self.model(records)

        if torch.isnan(student_scores).any() or torch.isinf(student_scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (pointwise distillation)."
            )
            sys.exit(1)

        teacher_scores = teacher_scores.to(student_scores.device)
        self.lossfn.process(student_scores, teacher_scores, self.context)
