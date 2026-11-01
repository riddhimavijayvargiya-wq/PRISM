import os
import re

CORPUS_DIR = "corpus"

def _extract_version(filepath: str) -> tuple:
    """Reads the document's own stated version (e.g. 'Version: 2.6' near
    the top of the file) rather than trusting the filename encoding."""
    with open(filepath, "r", encoding="utf-8") as f:
        head = f.read(500)

    # Require the match to start AND end on a digit, so trailing punctuation
    # (like a sentence-ending period right after the version number) never
    # gets swept into the captured group.
    match = re.search(r"Version:\s*(\d+(?:\.\d+)*)", head)
    if not match:
        return (0,)

    parts = [p for p in match.group(1).split(".") if p]
    return tuple(int(p) for p in parts) if parts else (0,)

def get_latest_version_files(corpus_dir: str = CORPUS_DIR) -> list[str]:
    """Groups files by document identity (e.g. 'pytorch__lstm', ignoring the
    version-specific filename suffix) and keeps only the highest-version
    file per group. Returns filenames only, not full paths."""
    groups = {}

    for filename in os.listdir(corpus_dir):
        if not filename.endswith(".md"):
            continue

        parts = filename.split("__")
        if len(parts) < 2:
            continue
        doc_key = "__".join(parts[:2])  # "pytorch__lstm" — the document's identity

        filepath = os.path.join(corpus_dir, filename)
        version = _extract_version(filepath)

        if doc_key not in groups or version > groups[doc_key][0]:
            groups[doc_key] = (version, filename)

    return [filename for _, filename in groups.values()]