# Cost per 1M tokens, USD. All zero for now since every model is on the free tier —
# this table is what makes cost tracking "just work" later if you add a paid model.
MODEL_PRICING = {
    "gemini-3.6-flash": {"input": 0.0, "output": 0.0},
    "gemini-3.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-3.5-flash-lite": {"input": 0.0, "output": 0.0},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)