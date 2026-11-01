def source_diversity(sources: list[str]) -> int:
    """Count of distinct source documents among retrieved chunks — the
    metric that would have caught the version-crowding bug numerically,
    instead of needing manual inspection to notice it."""
    return len(set(sources))


def context_volume(chunks: list[str]) -> int:
    """Total word count across all retrieved chunks — a free proxy for
    how much material a pipeline handed the LLM to work with."""
    return sum(len(chunk.split()) for chunk in chunks)