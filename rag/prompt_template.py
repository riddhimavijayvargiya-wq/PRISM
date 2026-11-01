def build_rag_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so explicitly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""