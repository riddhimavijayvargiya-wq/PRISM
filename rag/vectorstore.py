import chromadb

_client = chromadb.PersistentClient(path="data/chroma")

def get_collection(name: str):
    return _client.get_or_create_collection(name)

def add_chunks(collection_name, chunks, embeddings, metadatas, ids):
    collection = get_collection(collection_name)
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

def query_collection(collection_name, query_embedding, top_k):
    collection = get_collection(collection_name)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
def reset_collection(name: str):
    """Deletes a collection entirely so re-ingestion starts clean, instead
    of adding new chunks on top of old (now-outdated) ones."""
    try:
        _client.delete_collection(name)
    except Exception:
        pass  # collection didn't exist yet — nothing to reset, that's fine

def get_all_documents(collection_name: str):
    """Pulls back everything stored in a collection — all chunks, their
    metadata, their IDs. Used by Pipeline C to build its keyword (BM25)
    index from the exact same chunks already stored for dense search,
    rather than maintaining a second, separate copy of the corpus."""
    collection = get_collection(collection_name)
    return collection.get()