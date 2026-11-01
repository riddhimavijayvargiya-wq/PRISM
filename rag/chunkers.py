def chunk_text(text: str, chunk_size: int = 250, overlap: int = 0) -> list[str]:
    """Word-based chunking (a simple approximation of token count — close
    enough for this project; a real tokenizer would count more precisely)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks