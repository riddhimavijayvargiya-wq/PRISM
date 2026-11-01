import os
import threading
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from rag.chunkers import chunk_text
from rag.embeddings import embed
from rag.vectorstore import add_chunks, query_collection, reset_collection, get_all_documents
from rag.corpus_utils import get_latest_version_files

COLLECTION_NAME = "pipeline_c"
CHUNK_SIZE = 450
OVERLAP = 75
CANDIDATE_K = 10   # how many each retrieval method contributes, before reranking
FINAL_K = 3        # how many survive after reranking
CORPUS_DIR = "corpus"

# Loaded once at import time, same pattern as the embedding model in
# rag/embeddings.py — loading a model is expensive, so every pipeline
# does it exactly once per process, not once per question.
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
_rerank_lock = threading.Lock()

# BM25 needs to hold the whole corpus's tokenized text in memory to score
# against — built lazily on first use, then cached for the rest of the run.
_bm25_index = None
_bm25_chunks = None
_bm25_metadatas = None


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
    print(f"Pipeline C: ingested {len(chunks)} chunks from {len(filenames)} documents")


def _load_bm25():
    global _bm25_index, _bm25_chunks, _bm25_metadatas
    if _bm25_index is not None:
        return  # already built this run

    data = get_all_documents(COLLECTION_NAME)
    _bm25_chunks = data["documents"]
    _bm25_metadatas = data["metadatas"]

    tokenized = [doc.lower().split() for doc in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized)

def _mmr_select(candidate_docs, candidate_metas, candidate_embeddings, relevance_scores, final_k, lambda_param=0.7):
    """Maximal Marginal Relevance: iteratively picks whichever remaining
    candidate best balances relevance to the query against redundancy
    with chunks already selected — so a highly relevant but repetitive
    chunk gets penalized once something similar is already in the set."""
    embeddings = np.array(candidate_embeddings)

    # Normalize reranker scores to 0-1 so they're on a comparable scale to
    # cosine similarity — without this, whichever term has the larger raw
    # numeric range would silently dominate the balance.
    min_r, max_r = min(relevance_scores), max(relevance_scores)
    norm_relevance = [(r - min_r) / (max_r - min_r + 1e-9) for r in relevance_scores]

    selected, remaining = [], list(range(len(candidate_docs)))

    while remaining and len(selected) < final_k:
        best_i, best_score = None, float("-inf")

        for i in remaining:
            relevance = norm_relevance[i]
            if selected:
                # Cosine similarity = dot product, since embeddings are
                # already normalized in rag/embeddings.py
                max_sim = max(float(np.dot(embeddings[i], embeddings[j])) for j in selected)
            else:
                max_sim = 0.0  # first pick — nothing to be redundant with yet

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            # print("i =", i)
            # print("relevance =", relevance)
            # print("max_sim =", max_sim)
            # print("mmr_score =", mmr_score)
            # print("best_score =", best_score)
            if mmr_score > best_score:
                best_score, best_i = mmr_score, i

        selected.append(best_i)
        # print("remaining =", remaining)
        # print("best_i =", best_i)
        remaining.remove(best_i)

    return [(candidate_docs[i], candidate_metas[i], relevance_scores[i]) for i in selected]

def retrieve(question: str):
    _load_bm25()

    # --- Dense (semantic/embedding) candidates ---
    query_embedding = embed([question])[0]
    dense_results = query_collection(COLLECTION_NAME, query_embedding, CANDIDATE_K)
    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    # --- Keyword (BM25) candidates ---
    tokenized_query = question.lower().split()
    bm25_scores = _bm25_index.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_K]
    bm25_docs = [_bm25_chunks[i] for i in top_bm25_idx]
    bm25_metas = [_bm25_metadatas[i] for i in top_bm25_idx]

    # --- Union the two candidate sets, de-duplicated ---
    candidates = {}
    for doc, meta in zip(dense_docs, dense_metas):
        candidates[doc] = meta
    for doc, meta in zip(bm25_docs, bm25_metas):
        candidates[doc] = meta

    candidate_docs = list(candidates.keys())
    candidate_metas = list(candidates.values())

    # --- Rerank the combined pool, keep the true top 3 ---
    pairs = [[question, doc] for doc in candidate_docs]
    with _rerank_lock:
        scores = _reranker.predict(pairs)

    

    candidate_embeddings = embed(candidate_docs)
    top = _mmr_select(candidate_docs, candidate_metas, candidate_embeddings, scores, FINAL_K, lambda_param=0.7)

    return {
        "documents": [[doc for doc, _, _ in top]],
        "metadatas": [[meta for _, meta, _ in top]],
    }