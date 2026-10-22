import os
import pytest
from datamaestro import prepare_dataset
from datamaestro_ir.config.edu.stanford.lotte import LotteData


@pytest.mark.skip(reason="Requires downloading 3.5GB LOTTE dataset")
def test_lotte_dataset_structure():
    print(f"LotteData.DATA.path: {LotteData.DATA.path}")
    print(
        f"LotteData resources: {[r.name for r in LotteData.__dataset__.ordered_resources]}"
    )
    lotte_writing = prepare_dataset(
        "edu.stanford.lotte.recreation.test.search"
    ).instance()
    print(f"LotteData.files state: {LotteData.files.state}")
    print(f"Document store path: {lotte_writing.documents.path}")
    print(next(lotte_writing.documents.iter_documents()))
    print(f"Topics path: {lotte_writing.topics.path}")
    print(f"Topics file exists: {os.path.exists(lotte_writing.topics.path)}")
    print(next(lotte_writing.topics.iter()))
