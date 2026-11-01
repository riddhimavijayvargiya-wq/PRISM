from gateway.gateway import call_llm

MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
prompt = "Say hello in one sentence."

for model in MODELS:
    result = call_llm(model, prompt)
    print(f"\n--- {model} ---")
    print(f"Answer: {result['text']}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Tokens: in={result['input_tokens']} out={result['output_tokens']}")
    print(f"Cost: ${result['cost_usd']}")

   #to run - python -m scripts.test_gateway 