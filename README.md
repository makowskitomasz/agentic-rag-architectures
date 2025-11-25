# agentic-rag-architectures

## Environment configuration

API keys, provider selection, and model overrides are read from a local `.env` file. Create one by copying the example file:

```bash
cp .env.example .env
```

Available variables:

- `OPENAI_API_KEY`, `GEMINI_API_KEY` – required for the respective providers.
- `EMBEDDING_PROVIDER`, `LLM_PROVIDER` – pick `openai` or `gemini` independently for embeddings and the LLM.
- `OPENAI_EMBEDDING_MODEL`, `OPENAI_LLM_MODEL`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_LLM_MODEL` – optional per-provider model overrides. Defaults are applied when omitted.

All modules load `.env` via `python-dotenv` inside `src/config.py`, ensuring a single source of truth for API keys and model choices.
