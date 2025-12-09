# Agentic RAG Architectures

This repository explores advanced Retrieval-Augmented Generation (RAG) strategies built on a shared retrieval stack (hybrid dense + lexical search, reranking, cross-encoder re-ranking, optional query rewriting). On top of the baseline pipeline we compose several agents:

- **Vanilla RAG** – single-shot retrieval + generation.
- **Self-reflective RAG** – initial answer → structured critique → refinement.
- **Query Decomposition RAG** – planner splits multi-hop questions into sub-queries, aggregator stitches the answers.
- **Chain-of-Verification** – extracts factual statements, retrieves fresh evidence, labels each statement, and rewrites the answer.
- **Active Retrieval** – iteratively rewrites queries until sufficiency is met.
- **MARAG** – multi-role pipeline (Researcher → Analyst → Synthesizer) that exposes bullets and evidence trails.
- **MADAM-RAG** – two debaters plus a moderator that can ask follow-up questions and declare a winner or merged answer.
- **Routing Agent** – meta-planner selecting the best pipeline and embedding profile per question.

The project ships synthetic data (policy timelines + sport knowledge), notebooks for experiments/evaluation, tests, and documentation describing each architecture (`docs/architectures.md`) plus the artificial corpus (`docs/artificial_dataset.md`).

---

## Repository Highlights

- Hybrid retriever with optional reranking and cross-encoder re-ranking.
- Structured outputs powered by `pydantic` + `instructor` integration in `llm_orchestrator`.
- Modular agents living under `src/agents/` with shared utilities and logging (custom `logger.success` and blue LLM logs).
- Experiments notebook (`notebooks/03_experiments.ipynb`) that runs every architecture, computes grounding/semantic metrics, and saves `results/experiment_results.{csv,json}`.
- Evaluation dashboard (`notebooks/04_evaluation.ipynb`) that renders Plotly-dark charts, routing diagnostics, efficiency metrics, and exports `results/evaluation_summary.json`.
- Unit tests (e.g., `tests/test_active_retrieval.py`) ensure key agents behave deterministically.

---

## Prerequisites

- Python 3.11+ (the `environment.yml` targets Conda but you can also use `venv`).
- An OpenAI API key (and optionally Gemini) for embeddings/LLMs.
- `make`, `pip`, and a working C/C++ toolchain if you plan to install extra packages.

---

## Installation & Environment Setup

```bash
git clone https://github.com/<your-org>/agentic-rag-architectures.git
cd agentic-rag-architectures

# Create the Conda env (recommended)
conda env create -f environment.yml
conda activate ara

# OR use pip/venv
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt  # generated from environment.yml if needed
```

Copy the environment template and fill in your keys/models:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY, provider choices, etc.
```

All modules load `.env` via `src/config.py`, so once the variables are set you can run notebooks, scripts, or tests without additional configuration.

---

## Running the Pipelines

### 1. Generate embeddings/chunks (if necessary)
If you modify `data/raw/` or add new questions, regenerate processed artifacts (script not included in repo snapshot; adapt to your workflow).

### 2. Experiments notebook
Open `notebooks/03_experiments.ipynb` and run all cells. The notebook:

1. Loads processed chunks/embeddings.
2. Defines `ARCHITECTURE_RUNNERS` for every agent.
3. Runs the evaluation questions, capturing latency, token usage, keyword/semantic metrics, grounding score, and routing metadata.
4. Saves outputs in `results/experiment_results.csv`, `results/experiment_results.json`, and `results/experiment_logs.json`.

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

## Testing

```bash
pytest
```

Add `PYTHONPATH=.` if your environment requires it. Remember to append `sys.path` in new test files (see `tests/test_active_retrieval.py`) when running from notebook-friendly contexts.

---

## Documentation

- `docs/architectures.md` – step-by-step description of every agentic pipeline, structured outputs, and example flows.
- `docs/artificial_dataset.md` – explains the handcrafted knowledge base (policy narratives + sports files) and why the dataset stresses multi-hop, contradiction detection, and routing.
