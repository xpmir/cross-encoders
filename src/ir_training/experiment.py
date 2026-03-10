"""One experiment to rule them all, merge all possible configurations here"""

import logging
import shutil
import yaml
from functools import partial
from pathlib import Path
from attrs import asdict
import numpy as np
import pandas as pd
from jinja2 import Template

from experimaestro import setmeta
from experimaestro.annotations import tags as get_tags
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.configuration import FabricConfiguration
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
from xpmir.neural.huggingface import HFCrossScorer, hf_cross_scorer
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.distillation.listwise import (
    ADR_MSE,
    DistillRankNetLoss,
    DistillationListwiseTrainer,
    ListwiseSoftmaxCrossEntropy,
)
from xpmir.letor.distillation.pairwise import DistillationPairwiseTrainer, MSEDifferenceLoss
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener
from xpmir.neural.splade import splade_encoder_from_pretrained_hf


from configuration import Losses, CE_FineTuning, Validation, generate_grid
from tests import build_tests
from format import dataframe_to_latex, aggregation_hf
from validations import nano_msmarco_validation_datasets, nanobeir_validation_datasets
from index_utils import get_splade_index

logging.basicConfig(level=logging.INFO)


def get_task_by_tags(tasks:list, tags:dict):
    """Return the first task in tasks that has all the given tags."""
    for task in tasks:
        task_tags = get_tags(task)
        logging.debug(f"Checking task with tags {task_tags} against {tags}")
        if all(str(task_tags.get(tag)) == str(value) for tag, value in tags.items()):
            return task
    return None

def get_model_based_retrievers(cfg: CE_FineTuning):
    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        # batcher=PowerAdaptativeBatcher.C(),
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
            # batcher=PowerAdaptativeBatcher.C(),
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
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
        )


    ### Pairwise distillation losses ###
    elif loss_member is Losses.marginMSE:
        # define the trainer for monomlm
        return DistillationPairwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    ### Listwise losses ###
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
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logging.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logging.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 100 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=ADR_MSE.C(),
        )
    ## Not using this one
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
            # batcher=PowerAdaptativeBatcher.C(),
            batch_size=batch_size,
            hooks=[],
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
    learners = []
    all_weights = []
    
    def run_one_config(
        helper: LearningExperimentHelper, cfg: CE_FineTuning, grid_search_id: str
    ):
        """Main process for Cross-encoder training"""

        # Setup indices and validation/test base retrievers
        model_based_retrievers = get_model_based_retrievers(cfg)

        if cfg.retriever:
            # We don't use BM25, but a given sparse retriever
            
            splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(cfg.retriever)

            def splade_retriever(
                name,
                encoder,
                topk,
                documents: Documents,
            ) -> Retriever.C:
                return (
                    SparseRetriever.C(
                        index=get_splade_index(
                            documents, 
                            splade_encoder=splade_encoder, 
                            indexation_cfg=cfg.indexation, 
                            launcher_index=launcher_index,
                            init_tasks=retriever_init_tasks
                        ),
                        topk=topk,
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
                    index=get_splade_index(
                        documents, 
                        splade_encoder=splade_encoder, 
                        indexation_cfg=cfg.indexation, 
                        launcher_index=launcher_index,
                        init_tasks=retriever_init_tasks
                    ),
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

            test_retrievers = partial(splade_retriever, retriever_tag, splade_encoder, cfg.retrieval.k)
        else:
            base_model = BM25.C()
            retriever_init_tasks = [] # no init task for BM25
            
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
                init_tasks=retriever_init_tasks,
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

        ce_trainer: LossTrainer = build_trainer(cfg)
        # Build the model
        scorer_model, scorer_hf_init_tasks = hf_cross_scorer(hf_id=cfg.base)
        scorer_model.tag("scorer", grid_search_id)
        

        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

                
            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            if cfg.learner.validation == Validation.MSMARCO.value:
                msmarco_validation = ValidationListener.C(
                        id="bestval",
                        dataset=ds_val,
                        retriever=model_based_retrievers(
                            documents=validation_documents,
                            retrievers=val_retrievers,
                            scorer=scorer_model,
                        ).tag("first_stage", retriever_tag),
                        validation_interval=cfg.learner.validation_interval,
                        metrics={"RR@10": True, "AP": False, "nDCG": False},
                    ).tag("validation", "msmarco")
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
                    ).tag("first_stage", retriever_tag)

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
                trainer=ce_trainer, # How to train the model
                model=scorer_model, # The model to train
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=(
                    [msmarco_validation]
                    if cfg.learner.validation == Validation.MSMARCO.value
                    else validations
                ),
                # The hook used for evaluation
                hooks=hooks,
                # fabric settings
                fabric_config= FabricConfiguration.C(
                    strategy=cfg.learner.strategy,
                    precision=cfg.learner.precision,
                    accelerator=cfg.learner.accelerator,
                )
            )
            learners.append(learner)

            # Submit job and link
            outputs = learner.submit(launcher=launcher_learner, init_tasks=retriever_init_tasks + scorer_hf_init_tasks)
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
                    ].tag("validation", name).tag("scorer", grid_search_id).tag("seed", seed).tag("first_stage", retriever_tag)
                    all_weights.append(load_model)
                    tests.evaluate_retriever(
                        partial(
                            model_based_retrievers,
                            scorer=scorer_model,
                            retrievers=test_retrievers,
                        ),
                        launcher_evaluate,
                        model_id=f"{grid_search_id}-{name}-{metric_name}-{seed}",
                        init_tasks=[load_model] + retriever_init_tasks,
                    )

    all_configs, all_tags = generate_grid(cfg)
    config_map = {}

    for config, cfg_tags in zip(all_configs, all_tags):        
        #just run the config
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        config_map[tagspath] = config
        logging.info(f"Running config with tags {tagspath}")
        run_one_config(helper=helper, cfg=config, grid_search_id=tagspath)

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    group_by_tags = [("tag", "first_stage"), ("tag", "scorer")]
    model_id_tags = group_by_tags + [("tag", "seed")]

    # 1. Convert to numeric
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

    output_file = helper.xp.resultspath / "raw_results.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Raw results (before aggregation) saved to {output_file}")

    # 1. compute the mean performance per scorer before aggregating (over seeds)
    # We compute the mean over all seeds and datasets for each model (scorer + seed + validation combo).
    mean_per_model = df.groupby(model_id_tags)[metric_cols].mean(numeric_only=True)


    # Read model card template
    template_path = Path(__file__).parent / "CrossEncoderCard.md"
    card_template_txt = template_path.read_text() if template_path.exists() else None

    def get_name_from_tags(model_tags:dict) -> str:
        """created the HF id from tags"""
        from format import loss_names, backbone_names_lower
        logging.debug(f"got tags {model_tags}")
        loss = model_tags.get('learner.loss')
        base = model_tags.get('base')
        #try to get prettier name
        loss = loss_names.get(loss,loss).replace("/","-")
        base = backbone_names_lower.get(base,base).replace("/","-")
        return f"cross-encoder-{base}-{loss}"
    
    # 2. Extract the best model for EACH scorer based on the mean nDCG@10
    if not mean_per_model.empty:
        models_path = helper.xp.resultspath / "models"
        if not models_path.exists():
            models_path.mkdir(parents=True, exist_ok=True)
        # Group by the configuration (everything except seed) to find the best seed for each
        # Or more simply, group by ('tag', 'scorer') to find the best (seed, validation) for each scorer
        best_models_indices = mean_per_model.groupby(("tag", "scorer"))[[("metric", "nDCG@10")]].idxmax()
        
        best_models_list = []

        for scorer_tagspath, best_model_row in best_models_indices.iterrows():
            # best_model_idx is the value in the first (and only) column
            best_model_idx = best_model_row.iloc[0]
            # best_model_idx is a tuple (first_stage, scorer, validation, seed)
            best_tags_cols = dict(zip(mean_per_model.index.names, best_model_idx))
            best_tags = {k[1]: v for k, v in best_tags_cols.items() if k[0] == "tag"}  # Keep only tag columns

            logging.info(f"Best model for scorer '{scorer_tagspath}' is: {best_tags}")

            # Filter the original dataframe to keep ALL rows (datasets) belonging to this best model
            mask = pd.Series(True, index=df.index)
            for tag_col, val in best_tags_cols.items():
                mask &= (df[tag_col].astype(str) == str(val))
            
            best_models_list.append(df[mask])

            # Print metrics in markdown
            best_model_results = df[mask].copy()
            
            # 1. Get results for this model, and flatten columns if needed
            if isinstance(best_model_results.columns, pd.MultiIndex):
                # Map tuples to simple names: ('metric', 'AP') -> 'AP', ('dataset', '') -> 'dataset'
                flat_cols = []
                for col in best_model_results.columns:
                    if col[0] == 'metric':
                        flat_cols.append(col[1])
                    elif col[0] == 'dataset':
                        flat_cols.append('dataset')
                    else:
                        flat_cols.append("_".join(str(x) for x in col if x))
                best_model_results.columns = flat_cols
            
            metrics_to_show = ["RR@10", "nDCG@10"]
            cols_to_keep = [c for c in ["dataset"] + metrics_to_show if c in best_model_results.columns]
            
            if cols_to_keep:
                best_model_results = best_model_results[cols_to_keep]
                
                # Add aggregations
                agg_names = []
                for agg_name, datasets in aggregation_hf.items():
                    mask_agg = best_model_results['dataset'].isin(datasets)
                    if mask_agg.any():
                        agg_row = best_model_results[mask_agg].mean(numeric_only=True).to_frame().T
                        agg_row['dataset'] = agg_name
                        best_model_results = pd.concat([best_model_results, agg_row], ignore_index=True)
                        agg_names.append(agg_name)
                
                # Format numeric columns: * 100
                numeric_cols = best_model_results.select_dtypes(include=[np.number]).columns
                best_model_results[numeric_cols] = (best_model_results[numeric_cols] * 100)
                
                # Create a version for display (Markdown) with bolding for aggregations
                best_model_results_md = best_model_results.copy()
                
                # Round for the CSV
                best_model_results[numeric_cols] = best_model_results[numeric_cols].round(2)
                
                # Format for Markdown: 2 decimal places and bold aggregations
                for col in numeric_cols:
                    best_model_results_md[col] = best_model_results_md[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else x
                    )

                for agg_name in agg_names:
                    idx = best_model_results_md[best_model_results_md['dataset'] == agg_name].index
                    best_model_results_md.loc[idx, 'dataset'] = f"**{agg_name}**"
                    for col in numeric_cols:
                        best_model_results_md.loc[idx, col] = best_model_results_md.loc[idx, col].apply(
                            lambda x: f"**{x}**" if pd.notna(x) else x
                        )

                logging.info(f"Results for best model '{scorer_tagspath}':\n{best_model_results_md.to_markdown(index=False)}")
            else:
                logging.warning(f"No metric columns found to display for best model '{scorer_tagspath}'")
                best_model_results_md = best_model_results # Fallback
            
            model_tags = {}
            for s in best_tags['scorer'].split("_"):
                try:
                    k, v = s.split("=")
                    model_tags[k] = v
                except:
                    pass
            
            #2. We got results, we save them in a separate folder for this model, so that we can easily access them later, and link the tensorboard logs to it as well.
            model_name = get_name_from_tags(model_tags)
            best_model_path = models_path / model_name
            
            if best_model_path.exists() and not best_model_path.is_dir():
                best_model_path.unlink()

            elif not best_model_path.exists():
                best_model_path.mkdir(parents=True, exist_ok=True)
            best_model_results.to_csv(best_model_path / "results.csv", index=False)
            logging.info(f"Best model '{scorer_tagspath}' results saved to {best_model_path / 'results.csv'}")

            #get learner for this model to get the path to the tensorboard logs
            best_model_learner = get_task_by_tags(learners, best_tags)
            if not best_model_learner:
                logging.warning(f"No learner found for best model '{scorer_tagspath}' with tags {best_tags}")
                continue

            # Job Path
            best_model_jobpath = best_model_learner.jobpath
            symlink_path = best_model_path / "job_logs"
            if symlink_path.exists():
                symlink_path.unlink()  # Remove existing symlink if it exists
            symlink_path.symlink_to(best_model_jobpath)
            
            # TensorBoard logs path
            tb_path = best_model_learner.logpath
            if not tb_path.exists():
                logging.warning(f"TensorBoard logs path {tb_path} does not exist for best model '{scorer_tagspath}'")
            else:
                logging.info(f"TensorBoard logs for best model '{scorer_tagspath}' are located at: {tb_path}")
                # Link the tensorboard logs to the best model folder
                tb_symlink_path = best_model_path / "tensorboard_logs"
                if tb_symlink_path.exists():
                    tb_symlink_path.unlink()  # Remove existing symlink if it exists
                tb_symlink_path.symlink_to(tb_path)


            
            # print the path to the model weights, save them in the best model folder, and link the tensorboard logs to it as well
            best_load_model = get_task_by_tags(all_weights, best_tags)
            if best_load_model:
                logging.info(f"Best model '{scorer_tagspath}' weights are located at: {best_load_model.path}")
                # copy the model weights to the best model folder
                shutil.copy(best_load_model.path, best_model_path / "model_weights.pt")
            else:                
                logging.warning(f"No model weights found for best model '{scorer_tagspath}' with tags {best_tags}")

            # Write model card and config
            if card_template_txt:
                template = Template(card_template_txt)
                best_cfg = config_map.get(scorer_tagspath)
                if best_cfg:
                    card = template.render(
                        base=best_cfg.base,
                        k=best_cfg.retrieval.k,
                        retriever=best_cfg.retriever if best_cfg.retriever else "BM25",
                        model_id=model_name, 
                        training_data="MS MARCO Passage", 
                        dataset="msmarco", 
                        loss= model_tags.get("learner.loss"),
                        results=best_model_results_md.to_markdown(index=False)
                        )
                    
                    with open(best_model_path / "README.md", "w") as f:
                        f.write(card)
                    logging.info(f"Model card written to {best_model_path / 'README.md'}")

                    # Save config as YAML
                    with open(best_model_path / "config.yaml", "w") as f:
                        yaml.dump(asdict(best_cfg.learner), f, default_flow_style=False)
                    logging.info(f"Configuration saved to {best_model_path / 'config.yaml'}")

        if best_models_list:
            best_model_df = pd.concat(best_models_list, ignore_index=True)

            # Save the evaluations for the best models (one per scorer)
            best_model_output_file = helper.xp.resultspath / "best_models_per_scorer_raw_results.csv"
            best_model_df.to_csv(best_model_output_file, index=False)
            logging.info(f"Best models (per scorer) raw results saved to {best_model_output_file}")


    # 2. Initial Grouping
    df_grouped = (
        df.groupby(["dataset"] + group_by_tags, dropna=False)[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )

    # Sort columns to avoid PerformanceWarning: indexing past lexsort depth
    df_grouped = df_grouped.sort_index(axis=1)

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
