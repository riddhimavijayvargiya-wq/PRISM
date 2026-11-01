import json
import re


def parse_json(raw_text: str):
    """Parses JSON from an LLM response, tolerating the markdown code
    fences models often add despite instructions. Returns None instead
    of raising, so each caller decides how to handle bad output."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None