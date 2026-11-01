import threading
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_lock = threading.Lock()

def embed(texts: list[str]) -> list[list[float]]:
    with _lock:
        return _model.encode(texts, normalize_embeddings=True).tolist()