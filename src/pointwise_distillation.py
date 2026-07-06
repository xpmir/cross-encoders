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


class PointwiseDistillationSampler(Sampler):
    """Sampler wrapper for pointwise distillation datasets"""

    samples: Param[list[PointwiseDistillationSamples]]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        class PointwiseSamplesDataset(ShardedIterableDataset):
            def __init__(self, samples_list):
                super().__init__()
                self.samples_list = samples_list

            def iter_shard(
                self, shard_id: int, num_shards: int
            ) -> Iterator[PointwiseDistillationSample]:
                idx = 0
                for samples in self.samples_list:
                    for sample in samples:
                        if idx % num_shards == shard_id:
                            yield sample
                        idx += 1

        return InfiniteDataset(PointwiseSamplesDataset(self.samples))


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
