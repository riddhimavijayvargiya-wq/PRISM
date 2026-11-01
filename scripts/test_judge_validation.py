from evaluation.judge import judge_answer

QUESTION = "What is the difference between an LSTM layer and a vanilla RNN layer in PyTorch?"

ASPECTS = [
    "An RNN computes only a hidden state, with a single equation per step",
    "An LSTM also keeps a cell state and uses input, forget, cell and output gates",
    "RNN has a nonlinearity parameter (tanh or relu); LSTM has a proj_size parameter",
    "RNN takes an initial hidden state; LSTM takes a tuple (h_0, c_0)",
]

CONTEXT = [
    "torch.nn.RNN applies a multi-layer Elman RNN with tanh or ReLU non-linearity. It computes a single hidden state h_t per time step: h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh). The nonlinearity parameter selects 'tanh' (default) or 'relu'. It takes an initial hidden state hx.",
    "torch.nn.LSTM applies a multi-layer long short-term memory RNN. For each element it computes input, forget, cell and output gates, a cell state c_t and a hidden state h_t. The proj_size parameter (default 0) enables LSTM with projections. It takes a tuple (h_0, c_0) of initial hidden and cell states.",
]

FULL = ("A torch.nn.RNN layer keeps a single hidden state h_t, updated each step with one tanh (or ReLU) equation. "
        "A torch.nn.LSTM layer additionally keeps a cell state c_t and computes input, forget, cell and output gates. "
        "RNN exposes a nonlinearity parameter ('tanh' or 'relu'), while LSTM has an optional proj_size parameter for projections. "
        "RNN takes an initial hidden state, whereas LSTM takes a tuple (h_0, c_0).")


def context_ok(e):
    return e["context_coverage"] is not None and e["context_coverage"] >= 0.75


CASES = {
    "good_full_answer": (
        FULL,
        lambda e: context_ok(e) and e["answer_coverage"] >= 0.75 and not e["refused"] and len(e["unsupported_claims"]) <= 1,
    ),
    "refusal": (
        "Based on the provided context, there is not enough information to answer this question.",
        lambda e: context_ok(e) and e["refused"] and e["false_refusal"] and e["answer_coverage"] <= 0.25,
    ),
    "off_topic": (
        "torch.nn.Conv2d applies a 2D convolution over an input image, with parameters such as in_channels, out_channels, kernel_size and stride.",
        lambda e: context_ok(e) and e["answer_coverage"] <= 0.25 and not e["refused"],
    ),
    "half_answer": (
        "An RNN keeps a single hidden state updated by one tanh equation. An LSTM also keeps a cell state and uses input, forget, cell and output gates.",
        lambda e: context_ok(e) and 0.25 <= e["answer_coverage"] <= 0.75 and (e["generation_efficiency"] or 0) <= 0.75,
    ),
    "invented_claim": (
        FULL + " LSTM also contains a built-in attention layer and a fifth 'memory gate' that RNN lacks.",
        lambda e: len(e["unsupported_claims"]) >= 1,
    ),
}

passed = 0
for name, (answer, check) in CASES.items():
    e = judge_answer(QUESTION, ASPECTS, CONTEXT, answer)
    if "evaluation_error" in e:
        print(f"ERROR {name}: {e['evaluation_error']}")
        continue
    ok = check(e)
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<18} ctx={e['context_coverage']} ans={e['answer_coverage']} "
          f"eff={e['generation_efficiency']} unsupported={len(e['unsupported_claims'])} "
          f"refused={e['refused']} false_refusal={e['false_refusal']}")
    print(f"      buckets: {[a['bucket'] for a in e['aspects']]}")

print(f"\n{passed}/{len(CASES)} cases behaved as expected")