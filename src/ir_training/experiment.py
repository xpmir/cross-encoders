"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial
import numpy as np
import pandas as pd
from transformers import AutoConfig

from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher

from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch import Random
from xpm_torch.batchers import PowerAdaptativeBatcher
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner

from xpmir.papers.helpers.samplers import (
    msmarco_v1_validation_dataset,
    prepare_collection,
    msmarco_hofstaetter_ensemble_hard_negatives,
)
import xpmir.interfaces.anserini as anserini
from xpmir.rankers.standard import BM25, Model
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.rankers import Documents, Retriever, document_cache, scorer_retriever

from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter

#TODO add support for those
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
    Sparse2BMPConverter,
)
from xpmir.text.adapters import TopicTextConverter
from xpmir.neural.splade import SpladeTextEncoderV2, MaxAggregation

# from xpmir.letor.distillation.pairwise import PairwiseTrainer, PointwiseCrossEntropyLoss

from configuration import Losses, CE_FineTuning, Validation
from tests import build_tests
from format import dataframe_to_latex
from validations import nanobeir_validation_datasets

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

    tests = build_tests(cfg.evaluation)

    # Setup indices and validation/test base retrievers
    model_based_retrievers = get_model_based_retrievers(cfg)

    if cfg.retriever:
        # We don't use BM25, but a given sparse retriever
        tokenizer = HFTokenizer.C(model_id=cfg.retriever)
        splade_encoder = SpladeTextEncoderV2.C(
            tokenizer=HFTokenizerAdapter.C(
                tokenizer=tokenizer, converter=TopicTextConverter.C()
            ),
            encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.retriever),
            aggregation=MaxAggregation.C(),
            maxlen=256,
        )

        @document_cache
        def splade_index(documents: Documents):
            logging.info(
                "Indexing %s (%s documents) with %s",
                documents.id,
                documents.count,
                launcher_index,
            )

            index = SparseRetrieverIndexBuilder.C(
                batch_size=cfg.indexation.batch_size,
                batcher=PowerAdaptativeBatcher.C(),
                encoder=splade_encoder,
                documents=documents,
                ordered_index=False,
                max_docs=cfg.indexation.max_indexed,
            ).submit(launcher=launcher_index)

            # Just submit the convertion for now
            Sparse2BMPConverter.C(
                index=index, block_size=32, compress_range=True
            ).submit(launcher=launcher_bmp)

            return index

        def splade_retriever(
            name,
            encoder,
            documents: Documents,
        ) -> Retriever.C:
            return (
                SparseRetriever.C(
                    index=splade_index()(documents),
                    topk=cfg.retrieval.k,
                    batchsize=1,
                    encoder=encoder,
                    in_memory=False,
                )
                .tag("first_stage", name)
                .tag("data", documents.id)
            )

        def splade_val_retrievers(
            documents: Documents,
            *,
            model: Model = None,
        ) -> Retriever.C:
            return SparseRetriever.C(
                index=splade_index()(documents),
                topk=cfg.learner.validation_top_k,
                batchsize=1,
                encoder=model,
                in_memory=True,
            )

        retriever_tag = cfg.retriever
        # Caches the Splade index task for a document collection
        val_retrievers_factory = partial(
            splade_val_retrievers,
            model=splade_encoder,
        )

        test_retrievers = partial(splade_retriever, retriever_tag, splade_encoder)
    else:
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

        val_retrievers_factory = partial(
            anserini.retriever,
            anserini.index_builder(launcher=launcher_index),
            model=base_model,
        )
        retriever_tag = "bm25"

        test_retrievers = partial(bm25_retriever, retriever_tag)

        # evaluate base retrievers alone
        tests.evaluate_retriever(
            test_retrievers,
            launcher=launcher_evaluate,
        )


    if cfg.learner.validation == Validation.MSMARCO.value:
        # Full MSMARCO validation dataset
        ds_val = msmarco_v1_validation_dataset(
            cfg.validation, launcher=launcher_preprocessing
        )

        if cfg.retriever:
            # We don't use BM25, but a given sparse retriever
            val_retrievers = val_retrievers_factory
        else:   
            val_retrievers = partial(
                val_retrievers_factory, store=train_documents, k=cfg.learner.validation_top_k
            )
            
    # NanoBEIR validation datasets
    elif cfg.learner.validation == Validation.NanoBEIR.value:
        validations, validation_documents = nanobeir_validation_datasets(
            cfg.validation, launcher=launcher_preprocessing
        )

        # Build a simple list of (name, validation_dataset, documents) for later use
        nb_val_items = [
            (name, validations[name], validation_documents[name])
            for name in validations.keys()
        ]

    else:
        raise NotImplementedError(
            f"Validation dataset {cfg.learner.validation} is not implemented yet."
        )

    ### TRAINING CROSS ENCODER
    for i in range(cfg.nb_repetitions):
        seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
        random = Random.C(seed=seed).tag("seed", seed)

        ce_trainer: LossTrainer = build_trainer(cfg)

        # Build the model
        
        config = AutoConfig.from_pretrained(cfg.base)
        scorer_model = HFCrossScorer.C(
            hf_id=cfg.base,
            max_length=config.max_position_embeddings,
        )
        scorer_model.tag("scorer", cfg.id)

        # The validation listener evaluates the full retriever
        # (retriever + scorer) and keep the best performing model
        # on the validation set
        if cfg.learner.validation == Validation.MSMARCO.value:
            validation = [
                ValidationListener.C(
                    id="bestval",
                    dataset=ds_val,
                    retriever=model_based_retrievers(
                        documents=train_documents,
                        retrievers=val_retrievers,
                        scorer=scorer_model,
                    ).tag("retriever", retriever_tag),
                    validation_interval=cfg.learner.validation_interval,
                    metrics={"RR@10": True, "AP": False, "nDCG": False},
                    ).tag("validation", "msmarco")
            ]
        elif cfg.learner.validation == Validation.NanoBEIR.value:
            validation = []
            for name, ds_val_zs, val_docs in nb_val_items:
                if cfg.retriever:
                    # We don't use BM25, but a given sparse retriever
                    val_retriever_ood = val_retrievers_factory
                else:   
                    val_retriever_ood = partial(
                        val_retrievers_factory, store=val_docs, k=cfg.learner.validation_top_k
                    )

                retriever = model_based_retrievers(
                    documents=val_docs,
                    retrievers=val_retriever_ood,
                    scorer=scorer_model,
                ).tag("retriever", retriever_tag)

                listener = ValidationListener.C(
                    id=f"bestval_zs_{name}",
                    dataset=ds_val_zs,
                    retriever=retriever,
                    validation_interval=cfg.learner.validation_interval,
                    metrics={"RR@10": True, "AP": False, "nDCG": False},
                )
                validation.append(listener)

            # Aggregator includes the main (if present) plus all per-dataset listeners
            aggregator_validation = AggregatorValidationListener.C(
                listeners=validation,
                id="aggregated_validation",
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            ).tag("validation", "nanobeir")
            validation.append(aggregator_validation)
        else:
            raise NotImplementedError(
                f"Validation dataset {cfg.learner.validation} is not implemented yet."
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
            listeners=validation,
            # The hook used for evaluation
            hooks=hooks,
            # fabric settings
            strategy=cfg.learner.strategy,
            precision=cfg.learner.precision,
            accelerator=cfg.learner.accelerator,
        )

        # Submit job and link
        outputs = learner.submit(launcher=launcher_learner)

        # If we track MSMARCO, use default validation, else (for NanoBEIR) use the aggregator 
        tracked_validation = validation[0] if cfg.learner.validation == Validation.MSMARCO.value else validation[-1]

        # Evaluate the neural model on test collections
        for metric_name in tracked_validation.monitored():
            load_model = outputs.listeners[tracked_validation.id][metric_name]
            # load_model = outputs.checkpoints[200]
            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer_model,
                    retrievers=test_retrievers,
                ),
                launcher_evaluate,
                model_id=f"{cfg.id}-{metric_name}-{seed}",
                init_tasks=[load_model],
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

    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    # Generate and save LaTeX table
    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=None,
    )
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")
