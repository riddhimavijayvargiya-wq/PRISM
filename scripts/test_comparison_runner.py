from api.api.comparison_runner import run_rag_playground, run_llm_playground

question = "What is the difference between an LSTM layer and a vanilla RNN layer in PyTorch, and why would you choose one over the other?"

print("=== RAG Playground: same model, vary pipeline ===")
results = run_rag_playground(question, model="gemini-3.6-flash")
for r in results:
    print(f"\n--- Pipeline {r['pipeline']} ---")
    if r["error"]:
        print(f"ERROR: {r['error']}")
    else:
        print(r["answer"][:300], "...")
        print(f"Sources: {r['sources']}")
        print(f"Total latency: {r['total_latency_ms']} ms | Cost: ${r['cost_usd']}")

print("\n\n=== LLM Playground: same pipeline, vary model ===")
results = run_llm_playground(
    question,
    pipeline_name="C",
    models=["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
)
for r in results:
    print(f"\n--- Model {r['model']} ---")
    if r["error"]:
        print(f"ERROR: {r['error']}")
    else:
        print(r["answer"][:300], "...")
        print(f"Total latency: {r['total_latency_ms']} ms | Cost: ${r['cost_usd']}")