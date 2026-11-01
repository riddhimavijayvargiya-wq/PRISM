import os
from rag.chunkers import chunk_text
from rag.embeddings import embed
from rag.vectorstore import add_chunks, query_collection, reset_collection
from rag.corpus_utils import get_latest_version_files

COLLECTION_NAME = "pipeline_b"
CHUNK_SIZE = 900
OVERLAP = 150
TOP_K = 4
CORPUS_DIR = "corpus"

def ingest():
    reset_collection(COLLECTION_NAME)

    ids, chunks, metadatas = [], [], []
    filenames = get_latest_version_files(CORPUS_DIR)

    for filename in filenames:
        with open(os.path.join(CORPUS_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, OVERLAP)):
            ids.append(f"{filename}::{i}")
            chunks.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    embeddings = embed(chunks)
    add_chunks(COLLECTION_NAME, chunks, embeddings, metadatas, ids)
    print(f"Pipeline B: ingested {len(chunks)} chunks from {len(filenames)} documents")

def retrieve(question: str):
    query_embedding = embed([question])[0]
    return query_collection(COLLECTION_NAME, query_embedding, TOP_K)