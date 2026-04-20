from datamaestro import prepare_dataset
from datamaestro_ir.config.edu.stanford.lotte import LotteData

print(f"LotteData.DATA.path: {LotteData.DATA.path}")
print(
    f"LotteData resources: {[r.name for r in LotteData.__dataset__.ordered_resources]}"
)
lotte_writing = prepare_dataset("edu.stanford.lotte.recreation.test.search").instance()
print(f"LotteData.files state: {LotteData.files.state}")
print(f"Document store path: {lotte_writing.documents.path}")
print(next(lotte_writing.documents.iter_documents()))
print(f"Topics path: {lotte_writing.topics.path}")
import os

print(f"Topics file exists: {os.path.exists(lotte_writing.topics.path)}")
print(next(lotte_writing.topics.iter()))
