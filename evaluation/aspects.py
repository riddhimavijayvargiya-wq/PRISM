from gateway.gateway import call_llm
from evaluation.json_utils import parse_json

# Only one call per comparison, so a stronger model is affordable here.
# If the checklist is poor, every score downstream is poor.
CHECKLIST_MODEL = "gemini-3.5-flash"

CHECKLIST_PROMPT = """You are helping evaluate answers to a technical question.

Question: {question}

List the distinct points that a complete, correct answer should cover.

Rules:
- 3 to 6 items, ordered by importance.
- Each item is one short, self-contained statement (under 20 words) of a fact or aspect, not a question.
- Base the list on what the question asks for, not on any particular document.
- Do not pad: if the question is narrow, use fewer items.

Respond with ONLY this JSON, no other text:
{{"aspects": ["...", "..."]}}
"""


def generate_checklist(question: str) -> list[str]:
    response = call_llm(CHECKLIST_MODEL, CHECKLIST_PROMPT.format(question=question))
    data = parse_json(response["text"])
    if not isinstance(data, dict) or not isinstance(data.get("aspects"), list):
        raise ValueError("Checklist response was not valid JSON")

    aspects = [a.strip() for a in data["aspects"] if isinstance(a, str) and a.strip()]
    if not 2 <= len(aspects) <= 8:
        raise ValueError(f"Checklist had {len(aspects)} items, expected 2-8")
    return aspects
