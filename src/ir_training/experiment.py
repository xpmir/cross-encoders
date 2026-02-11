"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial
from typing import Optional
import numpy as np
import pandas as pd
from transformers import AutoConfig

from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.utils.hugginface import prepare_hf_model
from xpm_torch.losses.batchwise import SoftmaxCrossEntropy
from xpm_torch.losses.pairwise import HingeLoss, PointwiseCrossEntropyLoss
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch.batchers import PowerAdaptativeBatcher
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.trainers.batchwise import BatchwiseTrainer
from xpm_torch.trainers.pairwise import PairwiseTrainer

from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_colbertv2_annotated,
    msmarco_rankdistillm_colbert_top50,
    msmarco_v1_validation_dataset,
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_efficient_sampler,
)
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.rankers.standard import BM25, Model
from xpmir.rankers import Documents, Retriever, scorer_retriever
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.distillation.listwise import (
    ADR_MSE,
    DistillRankNetLoss,
    DistillationListwiseTrainer,
    ListwiseSoftmaxCrossEntropy,
)
from xpmir.letor.distillation.pairwise import DistillationPairwiseTrainer, MSEDifferenceLoss
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter
from xpmir.text.adapters import TopicTextConverter
from xpmir.neural.splade import SpladeTextEncoderV2, MaxAggregation


from configuration import Losses, CE_FineTuning, Validation, generate_grid
from tests import build_tests
from format import dataframe_to_latex
from validations import nano_msmarco_validation_datasets, nanobeir_validation_datasets

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

    ### Pointwise losses
    if loss_member is Losses.BCE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=PointwiseCrossEntropyLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise losses ###
    elif loss_member is Losses.hingeLoss:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=HingeLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Listwise losses ###
    elif loss_member is Losses.infoNCE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        # Use the listwise distillation trainer for listwise-style losses.
        # Swap to a DistillationListwiseTrainer and a listwise distillation loss.
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch: 
            batch_size = int(np.sqrt(batch_size))
            passages_per_batch = batch_size * batch_size
            logging.warning(f"normalized batch size to {batch_size} to get {passages_per_batch} docs per batch")
        else:
            passages_per_batch = batch_size * batch_size
            logging.warning(f"Not normalizing docs per batch for InfoNCE, {batch_size}**2 docs = {passages_per_batch} docs per batch") 

        return BatchwiseTrainer.C(
            sampler=PairwiseInBatchNegativesSampler.C(
                sampler=msmarco_v1_docpairs_efficient_sampler(),
            ),
            lossfn=SoftmaxCrossEntropy.C(),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
            hooks=[],
        )

    elif loss_member is Losses.infoNCE_RankDistiLLM:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        # Use the listwise distillation trainer for listwise-style losses.
        # Swap to a DistillationListwiseTrainer and a listwise distillation loss.
        passages_per_query = 8 
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch: 
            batch_size = batch_size // passages_per_query
            logging.warning(f"normalized batch size to {batch_size} to get {batch_size * passages_per_query} docs per batch")
        else:
            logging.warning(f"Not normalizing docs per batch, {passages_per_query} docs x {batch_size} = {batch_size * passages_per_query} docs per batch") 

        return DistillationListwiseTrainer.C(
            sampler=msmarco_colbertv2_annotated(passages_per_query=passages_per_query),
            lossfn=ListwiseSoftmaxCrossEntropy.C(),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
        )

    ### Pairwise distillation losses ###
    elif loss_member is Losses.marginMSE:
        # define the trainer for monomlm
        return DistillationPairwiseTrainer.C(
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logging.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logging.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=ADR_MSE.C(),
        )
    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: CE_FineTuning):
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    #: Model-based retrievers
    train_documents = prepare_collection("irds.msmarco-passage.documents")

    tests = build_tests(cfg.evaluation)

    #cache the indexes 
    indexes = {}
    def get_splade_index(documents: Documents, splade_encoder, cfg: CE_FineTuning):
        """Build an index for given documents, using a given Sparse retriever model
        Caches it to avoid submitting job twice.
        """

        index_cfg = SparseRetrieverIndexBuilder.C(
            batch_size=cfg.indexation.batch_size,
            batcher=PowerAdaptativeBatcher.C(),
            encoder=splade_encoder,
            documents=documents,
            ordered_index=False,
            max_docs=cfg.indexation.max_indexed,
        )
        indexer_id = index_cfg.__identifier__()

        if indexer_id not in indexes:
            logging.info(
                "Indexing %s (%s documents) with %s",
                documents.id,
                documents.count,
                launcher_index,
            )
            index = index_cfg.submit(launcher=launcher_index)
            indexes[indexer_id] = index
        else:
            index = indexes[indexer_id]

        return index

    def run_one_config(
        helper: LearningExperimentHelper, cfg: CE_FineTuning, grid_search_id: str
    ):
        """Main process for Cross-encoder training"""

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


            def splade_retriever(
                name,
                encoder,
                documents: Documents,
            ) -> Retriever.C:
                return (
                    SparseRetriever.C(
                        index=get_splade_index(documents, splade_encoder=splade_encoder, cfg=cfg),
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
                    index=get_splade_index(documents, splade_encoder=splade_encoder, cfg=cfg),
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
                        index=anserini.index_builder(launcher=launcher_index)(
                            documents
                        ),
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

        ### Validation ###
        if cfg.learner.validation == Validation.MSMARCO.value:
            ds_val, validation_documents = nano_msmarco_validation_datasets(
                cfg.validation, launcher=launcher_preprocessing
            )

            if cfg.retriever:
                # We don't use BM25, but a given sparse retriever
                val_retrievers = val_retrievers_factory
            else:
                val_retrievers = partial(
                    val_retrievers_factory,
                    store=validation_documents,
                    k=cfg.learner.validation_top_k,
                )

        # NanoBEIR validation datasets
        elif cfg.learner.validation in [
            Validation.NanoBEIR.value,
            Validation.ALL.value,
        ]:
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

            prepare_hf_model(cfg.base)
            # Build the model

            config = AutoConfig.from_pretrained(cfg.base)
            scorer_model = HFCrossScorer.C(
                hf_id=cfg.base,
                max_length=config.max_position_embeddings,
                max_query_length=32,
                max_doc_length=256,
            )
            scorer_model.tag("scorer", grid_search_id)

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            if cfg.learner.validation == Validation.MSMARCO.value:
                msmarco_validation = [
                    ValidationListener.C(
                        id="bestval",
                        dataset=ds_val,
                        retriever=model_based_retrievers(
                            documents=validation_documents,
                            retrievers=val_retrievers,
                            scorer=scorer_model,
                        ).tag("retriever", retriever_tag),
                        validation_interval=cfg.learner.validation_interval,
                        metrics={"RR@10": True, "AP": False, "nDCG": False},
                    ).tag("validation", "msmarco")
                ]
            elif (
                cfg.learner.validation == Validation.NanoBEIR.value
                or cfg.learner.validation == Validation.ALL.value
            ):
                validations = []
                for name, ds_val_zs, val_docs in nb_val_items:
                    if cfg.retriever:
                        # We don't use BM25, but a given sparse retriever
                        val_retriever_ood = val_retrievers_factory
                    else:
                        val_retriever_ood = partial(
                            val_retrievers_factory,
                            store=val_docs,
                            k=cfg.learner.validation_top_k,
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
                    validations.append(listener)

                    if name == "msmarco":
                        msmarco_validation = listener

                # Aggregator includes the main (if present) plus all per-dataset listeners
                aggregator_validation = AggregatorValidationListener.C(
                    listeners=validations,
                    id="aggregated_validation",
                    validation_interval=cfg.learner.validation_interval,
                    metrics={"RR@10": True, "AP": False, "nDCG": False},
                )

                validations.append(aggregator_validation)
            else:
                raise NotImplementedError(
                    f"Validation dataset {cfg.learner.validation} is not implemented yet."
                )

            hooks = [setmeta(GradientLogHook.C(), True)]

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
                listeners=(
                    msmarco_validation
                    if cfg.learner.validation == Validation.MSMARCO.value
                    else validations
                ),
                # The hook used for evaluation
                hooks=hooks,
                # fabric settings
                strategy=cfg.learner.strategy,
                precision=cfg.learner.precision,
                accelerator=cfg.learner.accelerator,
            )

            # Submit job and link
            outputs = learner.submit(launcher=launcher_learner)
            # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
            helper.tensorboard_service.add(learner, learner.logpath)

            # If we track MSMARCO, use default validation, else (for NanoBEIR) use the aggregator
            tracked_validations = {}
            if cfg.learner.validation in [
                Validation.NanoBEIR.value,
                Validation.ALL.value,
            ]:
                # add nano_beir = aggregated of all validation listeners
                tracked_validations["nano-beir"] = validations[-1]
            if cfg.learner.validation in [
                Validation.MSMARCO.value,
                Validation.ALL.value,
            ]:
                # add msm validation
                tracked_validations["msmarco"] = msmarco_validation

            # Evaluate each model on test collections
            for name, tracked_validation in tracked_validations.items():
                logging.info(f"evaluating from validation: {name}")
                for metric_name in tracked_validation.monitored():
                    load_model = outputs.listeners[tracked_validation.id][
                        metric_name
                    ].tag("validation", name)
                    tests.evaluate_retriever(
                        partial(
                            model_based_retrievers,
                            scorer=scorer_model,
                            retrievers=test_retrievers,
                        ),
                        launcher_evaluate,
                        model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                        init_tasks=[load_model],
                    )

    all_configs, tagspaths = generate_grid(cfg)

    for config, tagspath in zip(all_configs, tagspaths):
        
        #TODO clean dirty fix
        
        #get loss 
        try:
            loss_member = Losses(config.learner.loss)
        except ValueError:
            raise ValueError(
                f"Unknown loss function: {config.learner.loss}. Accepted values are: {[e.value for e in Losses]}"
            )
        if loss_member is Losses.infoNCE_RankDistiLLM:
            logging.warning(f"running config with normalization")
            #run with normal config
            run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)
            config.normalize_docs_per_batch = True
            tagspath += "norm_size=True"
            #run with new config
            run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)
        elif loss_member is Losses.infoNCE:
            logging.warning(f"running config with normalization")
            config.normalize_docs_per_batch = True
            tagspath += "norm_size=True"
            #run with new config
            run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)
        else:
            #just run the config
            run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    group_by_tags = [("tag", "first_stage"), ("tag", "scorer"), ("tag", "validation")]

    # 1. Convert to numeric
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    # 2. Initial Grouping
    df_grouped = (
        df.groupby(["dataset"] + group_by_tags, dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )

    # 3. Add the 'mean' summary row
    if df_grouped["dataset"].nunique() > 1:
        # We aggregate the already aggregated means/vars
        # Note: Mean of means is mathematically sound;
        # Mean of vars is a common proxy for average instability.
        mean_df = (
            df_grouped.groupby(group_by_tags, dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )

        # Manually set the dataset label
        mean_df["dataset"] = "mean"

        # Ensure column order matches exactly before concat
        mean_df = mean_df[df_grouped.columns]
        df_grouped = pd.concat([df_grouped, mean_df], ignore_index=True)

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
