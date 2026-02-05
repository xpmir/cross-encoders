from typing import List
import logging
from attrs import Factory
from functools import partial
import pandas as pd
from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from transformers import AutoConfig


from experimaestro.launcherfinder import find_launcher

from xpm_torch.batchers import PowerAdaptativeBatcher
from xpm_torch.utils.hugginface import get_hf_config

from datamaestro_text.data.ir import Documents

from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2
from xpmir.papers import configuration
from xpmir.papers.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.rankers.standard import BM25
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, document_cache, Retriever
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.huggingface import (
    HFTokenizerAdapter,
    HFTokenizer
)
from xpmir.text.huggingface.base import HFMaskedLanguageModel

from format import dataframe_to_latex
from tests import minified_tests, paper_tests
from configuration import *


logging.basicConfig(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)

    scorers_hf_id: List[str] = []
    
    retrievers_hf_id: List[str] = [""]

    evaluation: Evaluation = Factory(Evaluation)

    retrievers_only: bool = False
    """If true, only evaluate first-stage retrievers without cross-encoders"""

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: BaselinesConfig
) -> PaperResults:

    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)

    if cfg.evaluation.all_datasets:
        tests = paper_tests(cfg.evaluation.test_max_topics, retrievers_only=cfg.retrievers_only)
    else:
        tests = minified_tests(cfg.evaluation.test_max_topics, retrievers_only=cfg.retrievers_only)

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
    ) #: Model-based retrievers

    ### BM25 Retriever

    @document_cache
    def index_builder(documents: Documents):
        return anserini.IndexCollection.C(
            documents=documents,
        ).submit(launcher=launcher_index,)

    def bm25Retriever(
        name,
        model,
        documents: Documents,
    ) -> Retriever.C:
        return anserini.AnseriniRetriever.C(
            index=index_builder()(documents),
            model=model,
            k=cfg.retrieval.k,
            store=documents,
        ).tag("first-stage", name)

    ### Build the retrievers list 
    retrievers = [partial(bm25Retriever, "bm25", BM25.C())]

    if len(cfg.retrievers_hf_id) > 0:
        for retriever_hf_id in cfg.retrievers_hf_id:
            logging.info(f"Instantiating retriever {retriever_hf_id} ")

            tokenizer = HFTokenizer.C(model_id=retriever_hf_id)
            splade_encoder = SpladeTextEncoderV2.C(
                tokenizer=HFTokenizerAdapter.C(
                    tokenizer=tokenizer, converter=TopicTextConverter.C()
                ),
                encoder=HFMaskedLanguageModel.from_pretrained_id(retriever_hf_id),
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

            retrievers.append(partial(splade_retriever, f"{retriever_hf_id}", splade_encoder))
    
    for retriever in retrievers:

        # Eval First stage only
        tests.evaluate_retriever(
            retriever,
            launcher=launcher_evaluate,
            )
        logging.info(f"First stage only evaluation done for {retriever}")
        logging.info(f"Evaluating model-based retrievers {cfg.scorers_hf_id}")
        # Eval With cross-encoder
        for scorer_hf_id in cfg.scorers_hf_id:
            # evaluating the zero-shot ability

            logging.info(f"Loading config for {scorer_hf_id} ")
            # TODO fix

            #config = get_hf_config(scorer_hf_id)
            config = AutoConfig.from_pretrained(scorer_hf_id)
            logging.info(config)

            scorer = HFCrossScorer.C(
                hf_id=scorer_hf_id,
                max_length=config.max_position_embeddings,
            )

            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer.tag("scorer", scorer_hf_id),
                    retrievers=retriever,
                ),
                launcher=launcher_evaluate,
                # model_id=f"{scorer_hf_id}", #need to be unique for each eval
                init_tasks=[],
            )

    helper.xp.wait()

    df = tests.to_dataframe()
    measures = ["AP", "RR@10", "nDCG@10"] if not cfg.retrievers_only else ["R@1000"]
    metric_cols = [("metric", measure) for measure in measures]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            ["dataset", ("tag", "first_stage"), ("tag", "scorer")] if ("tag", "scorer") in df.columns else ["dataset", ("tag", "first_stage")],
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
        metric_col="nDCG@10" if not cfg.retrievers_only else "R@1000",
    )
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")

