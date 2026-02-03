# Agentic RAG Architectures

## Project Map (Start Here)

If you are new to the repo, this is where things live and how to run it.

- **Core code:** `src/` (pipelines, agents, retrieval, logging).
- **Configs:** `configs/` (YAML experiment configs).
- **Datasets:** `data/` (synthetic `future_poland` + benchmark conversion).
- **Embeddings:** `embeddings/` (precomputed vectors per dataset).
- **Results:** `results/` (all outputs from experiments/single runs).
  - Typical files: `results/experiment_results.csv`, `results/experiment_logs.json`,
    `results/evaluation_summary.json`, `results/single_run.json`.
- **Notebooks:** `notebooks/` (experiments + evaluation dashboards).
- **Scripts:** `scripts/` (CLI runners, batch/single).
- **Docs:** `docs/` (architectures + dataset notes).
- **Tests:** `tests/` (unit tests for key agents).

**Quick run (end-to-end):**
1. `uv sync`
2. `cp .env.example .env` and set your API keys.
3. `uv run python scripts/run_experiments.py --config configs/experiment.yaml`
4. Open `notebooks/04_evaluation.ipynb` in Jupyter and run all cells.

The evaluation step writes summary artifacts to `results/`.

This repository explores advanced Retrieval-Augmented Generation (RAG) strategies built on a shared retrieval stack (hybrid dense + lexical search, reranking, cross-encoder re-ranking, optional query rewriting). On top of the baseline pipeline we compose several agents:

- **Vanilla RAG** – single-shot retrieval + generation.
- **Self-reflective RAG** – initial answer → structured critique → refinement.
- **Query Decomposition RAG** – planner splits multi-hop questions into sub-queries, aggregator stitches the answers.
- **Chain-of-Verification** – extracts factual statements, retrieves fresh evidence, labels each statement, and rewrites the answer.
- **Active Retrieval** – iteratively rewrites queries until sufficiency is met.
- **MARAG** – multi-role pipeline (Researcher → Analyst → Synthesizer) that exposes bullets and evidence trails.
- **MADAM-RAG** – two debaters plus a moderator that can ask follow-up questions and declare a winner or merged answer.
- **Routing Agent** – meta-planner selecting the best pipeline and embedding profile per question.

The project ships synthetic data (`future_poland`), a benchmark dataset conversion, notebooks for experiments/evaluation, tests, and documentation describing each architecture (`docs/architectures.md`) plus the artificial corpus (`docs/artificial_dataset.md`).

---

## Repository Highlights

- Hybrid retriever with optional reranking and cross-encoder re-ranking.
- Structured outputs powered by `pydantic` + `instructor` integration in `llm_orchestrator`.
- Modular agents living under `src/agents/` with shared utilities and logging (custom `logger.success` and blue LLM logs).
- Experiments notebook (`notebooks/03_experiments.ipynb`) that runs every architecture, computes grounding/semantic metrics, and saves results.
- Evaluation dashboard (`notebooks/04_evaluation.ipynb`) that renders Plotly-dark charts, routing diagnostics, efficiency metrics, and exports `results/evaluation_summary.json`.
- Unit tests (e.g., `tests/test_active_retrieval.py`) ensure key agents behave deterministically.

---

## Prerequisites

- Python 3.12+.
- `uv` for dependency management.
- An OpenAI API key (and optionally Gemini) for embeddings/LLMs.

---

## Installation & Environment Setup

```bash
git clone https://github.com/<your-org>/agentic-rag-architectures.git
cd agentic-rag-architectures

# Sync the uv environment
uv sync
```

Copy the environment template and fill in your keys/models:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY, provider choices, etc.
```

All modules load `.env` via `src/config.py`, so once the variables are set you can run notebooks, scripts, or tests without additional configuration.

---

## Data

- https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia - Miniwikipedia benchmark dataset
- https://huggingface.co/datasets/makotoma/future-poland-rag-benchmark - Future Poland dataset

---

## Data Layout

- `data/future_poland/raw/*.md` – synthetic knowledge base.
- `data/future_poland/processed/chunks.json` – chunked corpus.
- `data/future_poland/questions.csv` / `data/future_poland/d_questions.json` – evaluation questions.
- `data/benchmark/benchmark_files/*.md` – benchmark passages (single paragraphs).
- `data/benchmark/benchmark_questions.json` – benchmark questions.
- `embeddings/future_poland/*` – embeddings for the synthetic dataset.
- `embeddings/benchmark/*` – embeddings for the benchmark dataset.

---

## Running the Pipelines

### 1. Generate embeddings/chunks (if necessary)
Use the YAML runner with `flags.chunk` / `flags.embed` to rebuild chunks and embeddings.

### 2. Experiments notebook
Open `notebooks/03_experiments.ipynb` and run all cells. The notebook:

1. Loads processed chunks/embeddings.
2. Defines `ARCHITECTURE_RUNNERS` for every agent.
3. Runs the evaluation questions, capturing latency, token usage, keyword/semantic metrics, grounding score, and routing metadata.
4. Saves outputs in the configured `results/*` paths.

### 3. Evaluation dashboard
Open `notebooks/04_evaluation.ipynb`, run all cells. You’ll see:

- Summary tables per architecture.
- Latency/token/quality bar charts + distributions.
- Per-question win rates, trade-off rankings, and Pareto scatter plots.
- Routing diagnostics (pipeline/profile frequencies + heatmap).
- Advanced efficiency metrics and clustering view with non-overlapping labels.
- `results/evaluation_summary.json` is generated for downstream reporting.

### 4. Programmatic usage

You can call the main RAG pipeline directly:

```python
from src.rag_pipeline import rag

result = rag(
    query="Which sport has teams of 6 players: football or volleyball?",
    chunks=CHUNKS,
    embeddings=EMBEDDINGS,
    index_map=INDEX_MAP,
    provider="openai",
    llm_model="gpt-4o-mini",
    use_routing_agent=True,  # or select specific agent flags
)
print(result["answer"])
```

The pipeline exposes many switches (`use_chain_of_verification`, `use_active_retrieval`, `use_marag`, `use_madam_rag`, etc.) so you can run any architecture outside of the notebooks.

---

## Agents (summary)

- **vanilla** – single retrieval + answer.
- **self_reflective** – answer, critique, refine.
- **query_decomposition** – split multi-hop question into sub-queries, then aggregate.
- **chain_of_verification** – extract statements, verify with retrieval, rewrite.
- **active_retrieval** – iterative query rewriting until sufficient context.
- **marag** – Researcher → Analyst → Synthesizer.
- **madam_rag** – two debaters + moderator (follow-ups possible).
- **routing** – planner chooses best pipeline/profile per question.

---

## Question/Answer Format

The experiment scripts expect a **questions JSON** in this shape:

```json
[
  {
    "id": "Q001",
    "question": "Your question text",
    "answer": "Reference answer (optional)",
    "type": "optional category"
  }
]
```

Required:
- `question`

Optional (used for analysis/logging):
- `id`, `answer`, `type`

---

## Batch Experiments (YAML)

Use the YAML runner to execute all architectures and compute metrics:

```bash
uv run python scripts/run_experiments.py --config configs/experiment.yaml
```

What it does:
- Loads questions from the YAML config.
- Loads or builds chunks + embeddings (based on `flags.*`).
- Runs each requested architecture.
- Computes metrics (keyword/semantic precision+recall, grounding score).
- Writes `results/experiment_results.{csv,json}` and `results/experiment_logs.json`.

Notes:
- `experiment_results.csv` does **not** include `answer` (compact metrics table).
- `experiment_logs.json` includes `answer`, chunks, metadata, and errors.

Key config fields (see `configs/experiment.yaml`):
- `questions.*` – path + field names.
- `sources.raw_dir` – corpus for chunking.
- `flags.*` – enable/disable chunking/embedding.
- `pipelines.run` – architectures to execute.
- `run.*` – iterations, concurrency, reranker, checkpoints.

---

## Single-Question Runner (Interactive)

Interactive CLI for one question + one architecture:

```bash
uv run python scripts/run_single.py
```

What it does:
- Prompts for architecture and question text.
- Runs the chosen pipeline once.
- Prints JSON and writes `results/single_run.json` by default.

---

## Benchmark Run

Run the benchmark configuration (separate embeddings and outputs):

```bash
uv run python scripts/run_experiments.py --config configs/experiment_benchmark.yaml
```

---

## Testing

```bash
uv run pytest
```

Add `PYTHONPATH=.` if your environment requires it. Remember to append `sys.path` in new test files (see `tests/test_active_retrieval.py`) when running from notebook-friendly contexts.

---

## Documentation

- `docs/architectures.md` – step-by-step description of every agentic pipeline, structured outputs, and example flows.
- `docs/artificial_dataset.md` – explains the handcrafted knowledge base (policy narratives + sports files) and why the dataset stresses multi-hop, contradiction detection, and routing.
