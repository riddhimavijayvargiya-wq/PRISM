# Prism

**A RAG evaluation workbench: compare RAG pipelines and LLMs side by side, and see whether answer quality comes from retrieval or generation.**

Prism has two live comparison modes:

- **RAG Playground**: one LLM, several RAG pipelines. Isolates the effect of retrieval.
- **LLM Playground**: one RAG pipeline, several LLMs. Isolates the effect of the model.

Every answer is evaluated against a checklist of points a complete answer needs, so you can see *where* an answer fell short: the information was never retrieved, it was retrieved but not used, or the answer made claims the context doesn't support.

<!-- Add a screenshot: ![Prism RAG Playground](docs/screenshot.png) -->

## How evaluation works

My first evaluator asked an LLM judge for 1-5 scores (groundedness, relevance, completeness). It gave 5/5/5 to almost everything, including an answer that refused to respond, so I replaced it with a checklist-based design:

1. A checklist of 3-6 points a complete answer should cover is generated once per question, from the question alone.
2. For each answer, a judge labels every point twice: is it in the retrieved context, and is it in the answer?
3. Python classifies each point from those two labels:

| In context | In answer | Bucket |
|---|---|---|
| yes | yes | Covered |
| yes | no | Generation miss |
| no | no | Retrieval miss |
| no | yes | Unsupported |

**Metrics**
- Context coverage: checklist points present in the retrieved context (retrieval quality, no ground truth needed)
- Answer coverage: checklist points the answer states
- Generation efficiency: covered points divided by points available in context
- Unsupported claims: claims the context doesn't support
- Refusal flag, with "false refusal" when the model refused although the context had the information
- System metrics: latency (excludes judging), cost, source diversity, context size, error rate

The judge was validated on five synthetic cases (full answer, refusal, off-topic, half answer, invented claim) in `scripts/test_judge_validation.py`.

## The three RAG pipelines

| | A: Precision | B: Context-rich | C: Hybrid + rerank |
|---|---|---|---|
| Chunking | ~250 words, no overlap | ~900 words, 150 overlap | ~450 words, 75 overlap |
| Retrieval | dense, top 3 | dense, top 4 | dense + BM25, 10 candidates each |
| Final selection | top-k | top-k | cross-encoder rerank, then MMR to 3 |

They are deliberately different strategies rather than successive versions of each other, so which one wins depends on the question.

Shared components: `BAAI/bge-small-en-v1.5` embeddings (local), Chroma vector store (one collection per pipeline), `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker.

## Architecture

```
Frontend (HTML/JS) -> FastAPI -> Comparison runner (parallel threads)
                                   |-> RAG pipelines A / B / C -> Chroma, BM25
                                   |-> Gateway -> Gemini models (retry, latency, cost tracking)
                                   '-> Evaluator (checklist + judge)
```

## Tech stack
Python, FastAPI, ChromaDB, Sentence Transformers, rank-bm25, NumPy, google-genai (Gemini), tenacity, vanilla JavaScript (marked.js, KaTeX)

## Setup

```bash
git clone https://github.com/riddhimavijayvargiya-wq/PRISM.git
cd PRISM
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with a Gemini API key from https://aistudio.google.com/apikey:
```
GOOGLE_API_KEY=your_key_here
```

**Corpus.** The documents come from the gated Hugging Face dataset [Kajaluxan/version-aware-rag-corpus](https://huggingface.co/datasets/Kajaluxan/version-aware-rag-corpus) and are not redistributed here. Accept the dataset terms on Hugging Face, then:
```bash
huggingface-cli login
python -m scripts.download_corpus   # downloads the dataset, copies python, pytorch, kubernetes
python -m scripts.expand_corpus     # adds nodejs, pandas, golang, nvidia_tensorrt-llm, ros2
```
Only the latest version of each document is indexed (about 40 documents).

**Build the indexes**
```bash
python -m scripts.ingest_pipeline_a
python -m scripts.ingest_pipeline_b
python -m scripts.ingest_pipeline_c
```

**Run** (two terminals)
```bash
uvicorn api.main:app --reload --port 8000
```
```bash
cd frontend && python -m http.server 5500 --bind 127.0.0.1
```
Open http://localhost:5500.

Notes: model names (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`) are set in `gateway/pricing.py` and the frontend dropdowns, so update them if Google retires one. On the free tier I hit a limit of about 20 requests per day per model, and one RAG Playground comparison makes 7 calls (1 checklist, 3 answers, 3 judge calls).

## Project structure
```
api/          FastAPI app, routes, and the parallel comparison runner
gateway/      Gemini wrapper: retry with backoff, latency, token and cost tracking
rag/          Chunking, embeddings, vector store, pipelines A/B/C, prompt template
evaluation/   Checklist generation, judge, and free metrics
frontend/     Single-page UI for both playgrounds
scripts/      Corpus download, ingestion, and test scripts
```

## What I learned building it
- **Version duplicates crowded out other documents.** Near-identical chunks from many versions of the same page filled the top-k, so a cross-document question retrieved one topic only. Fix: index only the latest version of each document.
- **A reranker alone dropped a relevant source.** Candidates included the missing document, but the cross-encoder's top 3 were all one topic. Fix: Maximal Marginal Relevance (MMR) for the final selection.
- **Shared models are not thread-safe.** Running pipelines in parallel crashed the shared embedding model and reranker. Fix: locks around the calls.
- **LLM judges inflate scores.** A single 1-5 score per answer couldn't separate a refusal from a good answer. Fix: checklist-based labelling, validated on synthetic cases.

## Known limitations
- Gemini models only, and the free-tier quota limits how many comparisons can run per day.
- The checklist is LLM-generated, so a weak checklist weakens the metrics. It is shown in the UI so it can be inspected.
- The corpus is small (about 40 documents, latest versions only), so version-aware retrieval is not tested.
- There is no per-comparison timeout yet. If a call stalls or the quota runs out, the UI can stay on "Running..." (under investigation).
- Error rates are kept in memory and reset on restart, and results are not saved.
- Retrieval metrics against ground truth (recall@k, MRR) are not included yet.