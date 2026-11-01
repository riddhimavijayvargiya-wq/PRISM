from gateway.gateway import call_llm
from evaluation.json_utils import parse_json

JUDGE_MODEL = "gemini-3.5-flash-lite"  # if the validation test below fails, try gemini-3.5-flash

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator checking an AI-generated answer against a checklist.

Question:
{question}

Checklist of points a complete answer should cover:
{checklist}

Retrieved context (the only material the answer was supposed to use):
{context}

Answer being evaluated:
{answer}

For EACH checklist item, decide two things independently:
- in_context: does the retrieved context explicitly state this point? Quote the supporting words, or write "none".
- in_answer: does the answer explicitly state this point? Merely mentioning the topic is not enough. Quote the supporting words, or write "none".

Also:
- unsupported_claims: factual claims made in the answer that the retrieved context does not support (max 5, short). Empty list if none.
- refused: true only if the answer declines to answer the question as a whole. A caveat about one sub-part does not count.

Write the evidence BEFORE each true/false value. Respond with ONLY this JSON, no other text:
{{"items": [{{"id": 1, "context_evidence": "...", "in_context": true, "answer_evidence": "...", "in_answer": true}}], "unsupported_claims": [], "refused": false}}
"""

# (in_context, in_answer) -> bucket. Classification happens here in Python,
# not in the LLM, so it's deterministic and the judge only answers two
# narrow yes/no questions per item.
BUCKETS = {
    (True, True): "covered",
    (True, False): "generation_miss",
    (False, False): "retrieval_miss",
    (False, True): "unsupported",
}


def _as_bool(value) -> bool:
    # LLMs occasionally return "false" as a string, and bool("false") is
    # True, which would silently flip a label.
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def judge_answer(question: str, aspects: list[str], context_chunks: list[str], answer: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        checklist="\n".join(f"{i}. {a}" for i, a in enumerate(aspects, start=1)),
        context="\n\n---\n\n".join(context_chunks),
        answer=answer,
    )
    response = call_llm(JUDGE_MODEL, prompt)
    base = {"judge_latency_ms": response["latency_ms"], "judge_cost_usd": response["cost_usd"]}

    parsed = parse_json(response["text"])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        return {**base, "evaluation_error": "Judge returned unparseable output"}

    by_id = {}
    for item in parsed["items"]:
        if isinstance(item, dict):
            try:
                by_id[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue

    labeled = []
    for i, text in enumerate(aspects, start=1):
        item = by_id.get(i)
        if item is None:
            return {**base, "evaluation_error": f"Judge skipped checklist item {i}"}
        in_context = _as_bool(item.get("in_context"))
        in_answer = _as_bool(item.get("in_answer"))
        labeled.append({
            "text": text,
            "in_context": in_context,
            "in_answer": in_answer,
            "bucket": BUCKETS[(in_context, in_answer)],
            "context_evidence": str(item.get("context_evidence", "")),
            "answer_evidence": str(item.get("answer_evidence", "")),
        })

    n = len(labeled)
    in_context_count = sum(a["in_context"] for a in labeled)
    in_answer_count = sum(a["in_answer"] for a in labeled)
    covered_count = sum(a["bucket"] == "covered" for a in labeled)

    raw_claims = parsed.get("unsupported_claims")
    raw_claims = raw_claims if isinstance(raw_claims, list) else []
    unsupported = [c for c in raw_claims if isinstance(c, str) and c.strip()][:5]

    refused = _as_bool(parsed.get("refused", False))
    context_coverage = in_context_count / n

    return {
        **base,
        "aspects": labeled,
        "context_coverage": round(context_coverage, 2),            # retrieval quality
        "answer_coverage": round(in_answer_count / n, 2),          # what the answer states
        "generation_efficiency": (                                  # generation quality
            round(covered_count / in_context_count, 2) if in_context_count else None
        ),
        "unsupported_claims": unsupported,
        "refused": refused,
        "false_refusal": refused and context_coverage >= 0.5,       # refused despite having the info
    }