from typing import List
import logging
from attrs import Factory
from functools import partial
from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from transformers import AutoConfig


from experimaestro.launcherfinder import find_launcher

from xpm_torch.batchers import PowerAdaptativeBatcher

from datamaestro_text.data.ir import Documents

from xpmir.papers import configuration
from xpmir.papers.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.rankers.standard import BM25
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, document_cache, Retriever
from xpmir.text.huggingface import (
    HFTokenizerAdapter,
    HFTokenizer
)

from tests import minified_tests, paper_tests
from configuration import *


logging.basicConfig(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)

    scorers_hf_id: List[str] = ["cross-encoder/ms-marco-MiniLM-L12-v2"]
    
    retrievers_hf_id: List[str] = [""]

    evaluation: Evaluation = Factory(Evaluation)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: BaselinesConfig
) -> PaperResults:
    

    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)

    if cfg.evaluation.all_datasets:
        tests = paper_tests(cfg.evaluation.test_max_topics)
    else:
        tests = minified_tests(cfg.evaluation.test_max_topics)

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

            config = AutoConfig.from_pretrained(scorer_hf_id)
            scorer = HFCrossScorer.C(
                hf_id=scorer_hf_id,
                max_length=config.max_position_embeddings,
            )

            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer.tag("model", scorer_hf_id),
                    retrievers=retriever,
                ),
                launcher=launcher_evaluate,
                # model_id=f"{scorer_hf_id}", #need to be unique for each eval
                init_tasks=[],
            )


    return PaperResults(
        models={
            "minilm-zs-RR@10": scorer,
        }, 
        evaluations=tests,
        tb_logs=None, 
    )
