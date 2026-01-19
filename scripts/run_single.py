from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with `uv sync`.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_experiments import (  # noqa: E402
    _build_runners,
    _load_or_build_chunks,
    _load_or_build_embeddings,
    _resolve_path,
)
from src.embedder import generate_query_embedding  # noqa: E402
from src.metrics import (  # noqa: E402
    grounding_score,
    precision_recall_k,
    semantic_precision_recall_k,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _prompt_choice(options: list[str]) -> str:
    print("Choose the architecture:")
    for idx, name in enumerate(options, start=1):
        print(f"{idx}) {name}")
    while True:
        try:
            raw = input("Enter number: ").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid selection, try again.")
        except (ValueError, EOFError):
            print("Invalid selection, try again.")


def _prompt_question() -> str:
    while True:
        try:
            question = input("Enter your question: ").strip()
        except EOFError:
            question = ""
        if question:
            return question
        print("Question cannot be empty.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single architecture on one question.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    parser.add_argument("--out", default="results/single_run.json", help="JSON output path.")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_yaml(config_path)

    chunks = _load_or_build_chunks(config)
    embeddings, index_map = _load_or_build_embeddings(config, chunks)
    runners = _build_runners(config, chunks, embeddings, index_map)

    architecture = _prompt_choice(sorted(runners.keys()))
    question = _prompt_question()
    runner = runners[architecture]
    result = runner(question)
    chunks_out = result["chunks"]
    answer_text = result["answer"]

    kp, kr = precision_recall_k(
        query=question,
        retrieved_chunks=chunks_out,
        all_chunks=chunks,
        k=int(config.get("retrieval", {}).get("k", 5)),
    )
    if isinstance(answer_text, str) and answer_text.strip():
        answer_embedding = generate_query_embedding(
            answer_text,
            provider=config.get("providers", {}).get("embedding_provider"),
            model=config.get("providers", {}).get("embedding_model"),
        )
        sp, sr = semantic_precision_recall_k(
            answer_embedding=answer_embedding,
            retrieved_chunks=chunks_out,
            all_chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            k=int(config.get("retrieval", {}).get("k", 5)),
        )
    else:
        sp, sr = 0.0, 0.0

    output = {
        "question": question,
        "architecture": architecture,
        "answer": answer_text,
        "time_ms": result["time_ms"],
        "tokens": result["tokens"],
        "keyword_precision_k": kp,
        "keyword_recall_k": kr,
        "semantic_precision_k": sp,
        "semantic_recall_k": sr,
        "grounding_score": grounding_score(answer_text or "", chunks_out),
        "chunks": chunks_out,
        "metadata": result.get("metadata", {}),
    }

    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_json)

    out_path = _resolve_path(args.out) or Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_json, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
