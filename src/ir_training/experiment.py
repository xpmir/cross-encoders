"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial
from typing import List

from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher

import numpy as np
import pandas as pd
import xpmir.interfaces.anserini as anserini
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch import Random
from xpm_torch.batchers import PowerAdaptativeBatcher

from xpmir.papers.helpers.samplers import (
    msmarco_v1_docpairs_efficient_sampler,
    msmarco_v1_validation_dataset,
    prepare_collection,
    msmarco_hofstaetter_ensemble_hard_negatives,
)
from xpmir.rankers import scorer_retriever
from xpmir.rankers.standard import BM25, Model

#TODO add support for those
# from xpmir.index.sparse import (
#     SparseRetriever,
#     SparseRetrieverIndexBuilder,
#     Sparse2BMPConverter,
# )

from xpmir.rankers import Documents, Retriever, document_cache
from xpmir.neural.huggingface import HFCrossScorer
from stats import run_statistical_tests

from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)

from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener

# from xpmir.letor.distillation.pairwise import PairwiseTrainer, PointwiseCrossEntropyLoss


from configuration import Losses, CE_FineTuning
from tests import build_tests, minified_tests, nfcorpus_validation_dataset, paper_tests

logging.basicConfig(level=logging.INFO)


def get_model_based_retrievers(cfg: CE_FineTuning):
    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
    )  #: Model-based retrievers

    return model_based_retrievers


def build_trainer(cfg: CE_FineTuning) -> LossTrainer:
    try:
        loss_member = Losses(cfg.learner.loss)
    except ValueError:
        raise ValueError(
            f"Unknown loss function: {cfg.learner.loss}. Accepted values are: {[e.value for e in Losses]}"
        )

    if loss_member is Losses.marginMSE:
        # define the trainer for monomlm
        return DistillationPairwiseTrainer.C(
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    # TODO: name properly the BCE loss function in the configuration as well
    # elif loss_member is Losses.PointWiseMSE:
    #     launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    #     return PairwiseTrainer.C(
    #         lossfn=PointwiseCrossEntropyLoss.C(),
    #         sampler=msmarco_v1_docpairs_efficient_sampler(
    #             sample_rate=cfg.learner.sample_rate,
    #             sample_max=cfg.learner.sample_max,
    #             launcher=launcher_preprocessing,
    #         ),
    #         batcher=PowerAdaptativeBatcher.C(),
    #         batch_size=cfg.learner.optimization.batch_size,
    #     )

    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: CE_FineTuning):
    """MiniLM-v2 model training"""
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_bmp = find_launcher(cfg.indexation.sparse2bmp_requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    #: Model-based retrievers
    train_documents = prepare_collection("irds.msmarco-passage.documents")

    ds_val = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    val_documents = prepare_collection("irds.beir.nfcorpus.documents")
    ds_val_zs = nfcorpus_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    tests = build_tests(cfg.evaluation)

    # Setup indices and validation/test base retrievers
    model_based_retrievers = get_model_based_retrievers(cfg)

    # Only BM25 for now - will add SPLADE later
    base_model = BM25.C()

    def bm25_retriever(name, documents: Documents) -> Retriever.C:
        return (
            anserini.AnseriniRetriever.C(
                k=cfg.retrieval.k,
                model=base_model,
                index=anserini.index_builder(launcher=launcher_index)(documents),
                store=documents,
            )
            .tag("first_stage", name)
            .tag("data", documents.id)
        )

    retrievers = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )

    val_retrievers = partial(
        retrievers, store=train_documents, k=cfg.learner.validation_top_k
    )
    val_retrievers_zs = partial(
        retrievers, store=val_documents, k=cfg.learner.validation_top_k
    )

    retriever_tag = "bm25"
    test_retrievers = partial(bm25_retriever, retriever_tag)

    # evaluate base retrievers alone
    tests.evaluate_retriever(
        test_retrievers,
        launcher=launcher_evaluate,
    )

    ### TRAINING CROSS ENCODER
    for i in range(cfg.nb_repetitions):
        seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
        random = Random.C(seed=seed).tag("seed", seed)

        ce_trainer: LossTrainer = build_trainer(cfg)

        # Build the model
        scorer_model: FrankenCrossScorer = build_scorer_model(
            cfg, attn_patches=cfg.attn_patches
        )
        scorer_model.tag("scorer", cfg.id)

        # The validation listener evaluates the full retriever
        # (retriever + scorer) and keep the best performing model
        # on the validation set
        validation = ValidationListener.C(
            id="bestval",
            dataset=ds_val,
            retriever=model_based_retrievers(
                documents=train_documents,
                retrievers=val_retrievers,
                scorer=scorer_model,
                device=device,
            ).tag("retriever", retriever_tag),
            validation_interval=cfg.learner.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        validation_zs = ValidationListener.C(
            id="bestval_zs",
            dataset=ds_val_zs,
            retriever=model_based_retrievers(
                documents=val_documents,
                retrievers=val_retrievers_zs,
                scorer=scorer_model,
                device=device,
            ).tag("retriever", retriever_tag),
            validation_interval=cfg.learner.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        aggregator_validation = AggregatorValidationListener.C(
            listeners=[validation, validation_zs],
            id="aggregated_validation",
            validation_interval=cfg.learner.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        hooks = [
            # setmeta(DistributedHook.C(models=[scorer_model]), True),
            setmeta(GradientLogHook.C(), True),
        ]

        if cfg.learner.max_grad_norm > 0:
            gradient_clipping_hook = GradientClippingHook.C(
                max_norm=cfg.learner.max_grad_norm
            )
            hooks.append(gradient_clipping_hook)

        # The learner trains the model
        learner = Learner.C(
            # Misc settings
            random=random,
            # How to train the model
            trainer=ce_trainer,
            # The model to train
            model=scorer_model,
            # Optimization settings
            steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
            optimizers=cfg.learner.optimization.optimizer,
            max_epochs=cfg.learner.optimization.max_epochs,
            checkpoint_interval=cfg.learner.checkpoint_interval,
            # The listeners (here, for validation)
            listeners=[validation, validation_zs, aggregator_validation],
            # The hook used for evaluation
            hooks=hooks,
            # fabric settings
            strategy=cfg.learner.strategy,
            precision=cfg.learner.precision,
            accelerator=cfg.learner.accelerator,
        )

        # Submit job and link
        outputs = learner.submit(launcher=launcher_learner)

        # Evaluate the neural model on test collections
        for metric_name in validation.monitored():
            load_model = outputs.listeners[validation.id][metric_name]
            # load_model = outputs.checkpoints[200]
            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer_model,
                    retrievers=test_retrievers,
                    device=device,
                ),
                launcher_evaluate,
                model_id=f"{cfg.id}-{metric_name}-{seed}",
                init_tasks=[load_model],
            )

        ### Compute the baseline comparison if required ###
        if cfg.compare_with_baseline:
            # Build the baseline: pass an empty list of attention patches
            baseline_scorer_model: FrankenCrossScorer = build_scorer_model(
                cfg, attn_patches=[]
            )
            baseline_scorer_model.tag("scorer", f"baseline-{cfg.id}")

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validation_baseline = ValidationListener.C(
                id="bestval",
                dataset=ds_val,
                retriever=model_based_retrievers(
                    documents=train_documents,
                    retrievers=val_retrievers,
                    scorer=baseline_scorer_model,
                    device=device,
                ).tag("retriever", retriever_tag),
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            validation_zs_baseline = ValidationListener.C(
                id="bestval_zs",
                dataset=ds_val_zs,
                retriever=model_based_retrievers(
                    documents=val_documents,
                    retrievers=val_retrievers_zs,
                    scorer=baseline_scorer_model,
                    device=device,
                ).tag("retriever", retriever_tag),
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            aggregator_validation_baseline = AggregatorValidationListener.C(
                listeners=[validation_baseline, validation_zs_baseline],
                id="aggregated_validation",
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            # The learner trains the model
            baseline_learner = Learner.C(
                # Misc settings
                random=random,
                # How to train the model
                trainer=ce_trainer,
                # The model to train
                model=baseline_scorer_model,
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=[
                    validation_baseline,
                    validation_zs_baseline,
                    aggregator_validation_baseline,
                ],
                # The hook used for evaluation
                hooks=hooks,
                # fabric settings
                strategy=cfg.learner.strategy,
                precision=cfg.learner.precision,
                accelerator=cfg.learner.accelerator,
            )

            # Submit job and link
            baseline_outputs = baseline_learner.submit(launcher=launcher_learner)

            # Evaluate the neural model on test collections
            for metric_name in validation_baseline.monitored():
                load_baseline = baseline_outputs.listeners[validation_baseline.id][
                    metric_name
                ]
                # load_model = outputs.checkpoints[200]
                tests.evaluate_retriever(
                    partial(
                        model_based_retrievers,
                        scorer=baseline_scorer_model,
                        retrievers=test_retrievers,
                        device=device,
                    ),
                    launcher_evaluate,
                    model_id=f"baseline-{cfg.id}-{metric_name}-{seed}",
                    init_tasks=[load_baseline],
                )

        # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
        # the linking works only if the task was generated and scheduled by experimaestro, so that the learner.logpath is set.
        helper.tensorboard_service.add(learner, learner.logpath)

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()
    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            ["dataset", ("tag", "first_stage"), ("tag", "scorer")],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

    if cfg.compare_with_baseline:
        logging.info("Loading detailed results for each runs, across each setups")
        detailed_df = run_statistical_tests(
            tests.per_model, nb_repetitions=cfg.nb_repetitions
        )

        logging.info("Detailed results across each setups:")
        logging.info(detailed_df)

        detailed_output_file = (
            helper.xp.resultspath / "statistical_significance_results.csv"
        )
        detailed_df.to_csv(detailed_output_file, index=True)
        logging.info(
            f"Statistical significance results saved to {detailed_output_file}"
        )

    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    # Generate and save LaTeX table
    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=detailed_df if cfg.compare_with_baseline else None,
    )
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")
