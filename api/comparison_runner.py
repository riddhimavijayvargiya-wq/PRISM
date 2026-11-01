import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag import pipeline_a, pipeline_b, pipeline_c
from rag.prompt_template import build_rag_prompt
from gateway.gateway import call_llm
from evaluation.aspects import generate_checklist
from evaluation.judge import judge_answer
from evaluation.metrics import source_diversity, context_volume

PIPELINES = {"A": pipeline_a, "B": pipeline_b, "C": pipeline_c}

# --- Error-rate tracking (in memory, resets on restart) ---
_pipeline_stats = defaultdict(lambda: {"total": 0, "errors": 0})
_model_stats = defaultdict(lambda: {"total": 0, "errors": 0})
_stats_lock = threading.Lock()  # runs happen on several threads at once


def _record_call(pipeline_name: str, model: str, errored: bool):
    with _stats_lock:
        for stats, key in [(_pipeline_stats, pipeline_name), (_model_stats, model)]:
            stats[key]["total"] += 1
            if errored:
                stats[key]["errors"] += 1


def get_pipeline_error_rate(pipeline_name: str) -> float:
    with _stats_lock:
        s = _pipeline_stats[pipeline_name]
        return s["errors"] / s["total"] if s["total"] else 0.0


def get_model_error_rate(model: str) -> float:
    with _stats_lock:
        s = _model_stats[model]
        return s["errors"] / s["total"] if s["total"] else 0.0


# --- Evaluation fields every result carries, filled or empty ---
EVAL_KEYS = ["aspects", "context_coverage", "answer_coverage", "generation_efficiency",
             "unsupported_claims", "refused", "false_refusal", "evaluation_error"]

EMPTY_EVALUATION = {
    "aspects": [], "context_coverage": None, "answer_coverage": None,
    "generation_efficiency": None, "unsupported_claims": [], "refused": None,
    "false_refusal": None, "evaluation_error": None, "judge_cost_usd": 0.0,
}


def _make_checklist(question: str):
    """Never raises: returns None on failure so a flaky checklist call
    degrades evaluation instead of breaking the comparison."""
    try:
        return generate_checklist(question)
    except Exception:
        return None


def _evaluate(question, checklist_future, chunks, answer) -> dict:
    aspects = checklist_future.result()  # blocks only if the checklist isn't ready yet
    if not aspects:
        return {**EMPTY_EVALUATION, "evaluation_error": "Checklist generation failed"}
    try:
        evaluation = judge_answer(question, aspects, chunks, answer)
    except Exception as e:
        return {**EMPTY_EVALUATION, "evaluation_error": f"Judge call failed: {e}"}
    if "evaluation_error" in evaluation:
        return {**EMPTY_EVALUATION, "judge_cost_usd": evaluation["judge_cost_usd"],
                "evaluation_error": evaluation["evaluation_error"]}
    return {**EMPTY_EVALUATION, **evaluation, "evaluation_error": None}


def _failure_result(pipeline_name, model, start, message) -> dict:
    return {
        "pipeline": pipeline_name, "model": model, "answer": None, "sources": [],
        "llm_latency_ms": None,
        "total_latency_ms": round((time.time() - start) * 1000, 2),
        "cost_usd": 0.0, "source_diversity": None, "context_volume": None,
        "pipeline_error_rate": get_pipeline_error_rate(pipeline_name),
        "model_error_rate": get_model_error_rate(model),
        "error": message,
        **{k: EMPTY_EVALUATION[k] for k in EVAL_KEYS},
    }


def _run_one(pipeline_name: str, model: str, question: str, checklist_future) -> dict:
    start = time.time()
    try:
        retrieval = PIPELINES[pipeline_name].retrieve(question)
        chunks = retrieval["documents"][0]
        sources = [meta["source"] for meta in retrieval["metadatas"][0]]
        response = call_llm(model, build_rag_prompt(question, chunks))
    except Exception as e:
        _record_call(pipeline_name, model, errored=True)
        return _failure_result(pipeline_name, model, start, str(e))

    # Latency stops here: judging is evaluation overhead, not part of the
    # pipeline being compared, so it shouldn't inflate the number.
    total_latency_ms = round((time.time() - start) * 1000, 2)
    _record_call(pipeline_name, model, errored=False)

    # A judge failure (say, its daily quota running out) must not throw
    # away an answer that was generated successfully.
    evaluation = _evaluate(question, checklist_future, chunks, response["text"])

    return {
        "pipeline": pipeline_name,
        "model": model,
        "answer": response["text"],
        "sources": sources,
        "llm_latency_ms": response["latency_ms"],
        "total_latency_ms": total_latency_ms,
        # Excludes the one shared checklist call (free tier, $0 today)
        "cost_usd": response["cost_usd"] + evaluation["judge_cost_usd"],
        "source_diversity": source_diversity(sources),
        "context_volume": context_volume(chunks),
        "pipeline_error_rate": get_pipeline_error_rate(pipeline_name),
        "model_error_rate": get_model_error_rate(model),
        "error": None,
        **{k: evaluation[k] for k in EVAL_KEYS},
    }


def _run_parallel(tasks: list[tuple[str, str]], question: str) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=len(tasks) + 1) as executor:
        # The checklist is generated concurrently with the answers instead
        # of before them. Each task only waits on it after its own answer
        # is done, by which point it's almost always ready.
        checklist_future = executor.submit(_make_checklist, question)
        future_to_task = {
            executor.submit(_run_one, p, m, question, checklist_future): (p, m)
            for p, m in tasks
        }
        for future in as_completed(future_to_task):
            results.append(future.result())
    return results


def run_rag_playground(question: str, model: str, pipeline_names: list[str] = None) -> list[dict]:
    pipeline_names = pipeline_names or list(PIPELINES.keys())
    return _run_parallel([(p, model) for p in pipeline_names], question)


def run_llm_playground(question: str, pipeline_name: str, models: list[str]) -> list[dict]:
    return _run_parallel([(pipeline_name, m) for m in models], question)