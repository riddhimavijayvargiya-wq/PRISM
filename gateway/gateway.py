from gateway.providers import call_gemini
from gateway.pricing import calculate_cost

def call_llm(model: str, prompt: str) -> dict:
    result = call_gemini(model, prompt)
    result["cost_usd"] = calculate_cost(model, result["input_tokens"], result["output_tokens"])
    return result