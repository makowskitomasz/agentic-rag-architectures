from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - requires dependency install
    raise RuntimeError("PyYAML is required. Install with `uv sync`.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.utils import get_logger as _get_logger
except ImportError:  # pragma: no cover - fallback for non-package runs
    _get_logger = None

from src.chunking import chunk_documents  # noqa: E402
from src.embedder import generate_embeddings, generate_query_embedding  # noqa: E402
from src.load_data import load_markdown_files  # noqa: E402
from src.metrics import (  # noqa: E402
    estimate_tokens,
    grounding_score,
    precision_recall_k,
    semantic_precision_recall_k,
)
from src.rag_pipeline import rag  # noqa: E402
from src.agents.query_decomposition_rag import query_decomposition_rag  # noqa: E402
from src.agents.self_reflective_rag import self_reflect_rag  # noqa: E402


logger = _get_logger(__name__) if callable(_get_logger) else None
if logger is None:  # pragma: no cover - fallback logger
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("EXPERIMENTS")


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_questions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions_cfg = config.get("questions", {})
    path = _resolve_path(questions_cfg.get("path"))
    if path is None or not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    fmt = (questions_cfg.get("format") or path.suffix.lstrip(".")).lower()
    question_field = questions_cfg.get("question_field", "question")
    id_field = questions_cfg.get("id_field", "id")
    type_field = questions_cfg.get("type_field", "type")
    answer_field = questions_cfg.get("answer_field", "answer")
    limit = questions_cfg.get("limit")

    if fmt == "json":
        data = _load_json(path)
        if not isinstance(data, list):
            raise ValueError("JSON questions file must be a list of objects.")
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            question = item.get(question_field)
            if not question:
                continue
            records.append(
                {
                    "question": question,
                    "question_id": item.get(id_field),
                    "question_type": item.get(type_field),
                    "reference_answer": item.get(answer_field),
                }
            )
    elif fmt == "csv":
        df = pd.read_csv(path)
        if question_field not in df.columns:
            raise ValueError(f"CSV missing question field '{question_field}'.")
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "question": row.get(question_field),
                    "question_id": row.get(id_field),
                    "question_type": row.get(type_field),
                    "reference_answer": row.get(answer_field),
                }
            )
    else:
        raise ValueError(f"Unsupported questions format: {fmt}")

    if limit:
        records = records[: int(limit)]
    if not records:
        raise ValueError("No questions loaded from the configured file.")
    return records


def _load_or_build_chunks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = config.get("project", {})
    processed_dir = _resolve_path(data_cfg.get("processed_dir", "data/future_poland/processed"))
    chunks_path = _resolve_path(data_cfg.get("chunks_path", str(processed_dir / "chunks.json")))

    flags = config.get("flags", {})
    do_chunk = bool(flags.get("chunk", False))
    overwrite = bool(flags.get("overwrite_chunks", False))

    if chunks_path.exists() and not overwrite and not do_chunk:
        logger.info("Loading chunks from %s", chunks_path)
        return _load_json(chunks_path)

    if not do_chunk and not chunks_path.exists():
        raise FileNotFoundError("Chunks file missing. Enable flags.chunk or provide chunks_path.")

    raw_dir = config.get("sources", {}).get("raw_dir", "data/future_poland/raw")
    documents = load_markdown_files(str(_resolve_path(raw_dir)))
    chunk_size = int(config.get("chunking", {}).get("chunk_size", 400))
    overlap = int(config.get("chunking", {}).get("overlap", 50))
    logger.info("Chunking %d documents -> %s", len(documents), chunks_path)
    return chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        output_path=str(chunks_path),
    )


def _load_or_build_embeddings(
    config: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    data_cfg = config.get("project", {})
    embeddings_dir = _resolve_path(data_cfg.get("embeddings_dir", "embeddings"))
    embeddings_path = _resolve_path(data_cfg.get("embeddings_path", str(embeddings_dir / "embeddings.npy")))
    index_path = _resolve_path(data_cfg.get("index_path", str(embeddings_dir / "embedding_index.json")))

    flags = config.get("flags", {})
    do_embed = bool(flags.get("embed", False))
    overwrite = bool(flags.get("overwrite_embeddings", False))

    if embeddings_path.exists() and index_path.exists() and not overwrite and not do_embed:
        logger.info("Loading embeddings from %s", embeddings_path)
        return np.load(embeddings_path), _load_json(index_path)

    if not do_embed and (not embeddings_path.exists() or not index_path.exists()):
        raise FileNotFoundError("Embeddings missing. Enable flags.embed or provide embeddings_path/index_path.")

    provider_cfg = config.get("providers", {})
    provider = provider_cfg.get("embedding_provider")
    model = provider_cfg.get("embedding_model")
    batch_size = int(config.get("embedding", {}).get("batch_size", 16))

    logger.info("Generating embeddings -> %s", embeddings_dir)
    return generate_embeddings(
        chunks,
        provider=provider,
        model=model,
        batch_size=batch_size,
        output_dir=str(embeddings_dir),
    )


def _make_result(
    answer: str,
    chunks: List[Dict[str, Any]],
    time_ms: float,
    tokens: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seen: set[str] = set()
    chunk_list: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            chunk_list.append(chunk)
    return {
        "answer": answer,
        "chunks": chunk_list,
        "time_ms": float(time_ms),
        "tokens": float(tokens),
        "metadata": metadata or {},
    }


def _chunks_from_ids(chunk_ids: List[str], chunk_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    resolved = []
    for cid in chunk_ids:
        chunk = chunk_lookup.get(cid)
        if chunk:
            resolved.append(chunk)
    return resolved


def _build_runners(
    config: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
) -> Dict[str, Callable[[str], Dict[str, Any]]]:
    provider_cfg = config.get("providers", {})
    provider = provider_cfg.get("llm_provider", "openai")
    llm_model = provider_cfg.get("llm_model")
    embedding_provider = provider_cfg.get("embedding_provider", provider)
    embedding_model = provider_cfg.get("embedding_model")

    retrieval_cfg = config.get("retrieval", {})
    k = int(retrieval_cfg.get("k", 5))
    threshold = float(retrieval_cfg.get("threshold", 0.5))

    run_cfg = config.get("run", {})

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    def run_vanilla(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
        )
        answer = result.get("answer", "")
        return _make_result(answer, result.get("chunks", []), result.get("time_ms", 0.0), estimate_tokens(answer))

    def run_self_reflective(question: str) -> Dict[str, Any]:
        query_embedding = generate_query_embedding(question, provider=embedding_provider, model=embedding_model)
        result = self_reflect_rag(
            query=question,
            chunks=chunks,
            query_embedding=query_embedding,
            embeddings=embeddings,
            index_map=index_map,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            provider=provider,
            llm_model=llm_model,
            temperature=float(run_cfg.get("self_reflect_temperature", 1.0)),
            k=k,
            threshold=threshold,
            use_active_retrieval=bool(run_cfg.get("self_reflect_active_retrieval", False)),
        )
        timings = result.get("timings", {})
        time_ms = timings.get("total_ms", sum(timings.values()))
        refined_answer = result.get("refined_answer", "")
        return _make_result(
            refined_answer,
            result.get("retrieved_chunks", []),
            time_ms,
            estimate_tokens(refined_answer),
            metadata={"initial_answer": result.get("initial_answer", "")},
        )

    def run_query_decomposition(question: str) -> Dict[str, Any]:
        result = query_decomposition_rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            k=k,
            threshold=threshold,
            use_reranker=bool(run_cfg.get("use_reranker", True)),
            reranker_model=run_cfg.get("reranker_model"),
            rerank_top_k=run_cfg.get("rerank_top_k"),
        )
        answer = result.get("final_answer", "")
        sub_chunks: List[Dict[str, Any]] = []
        for sub in result.get("sub_answers", []):
            sub_chunks.extend(sub.get("retrieved_chunks", []))
        time_ms = result.get("timings", {}).get("total_ms", result.get("time_ms", 0.0))
        return _make_result(
            answer,
            sub_chunks,
            time_ms,
            estimate_tokens(answer),
            metadata={"plan": result.get("plan", {})},
        )

    def run_chain_verification(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
            use_chain_of_verification=True,
            verification_iterations=int(run_cfg.get("verification_iterations", 2)),
            verification_statements=int(run_cfg.get("verification_statements", 3)),
        )
        answer = result.get("answer", "")
        return _make_result(
            answer,
            result.get("chunks", []),
            result.get("time_ms", 0.0),
            estimate_tokens(answer),
            metadata={"verification": result.get("verification")},
        )

    def run_active_retrieval(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
            use_active_retrieval=True,
            active_iterations=int(run_cfg.get("active_iterations", 3)),
            active_sufficiency_threshold=float(run_cfg.get("active_sufficiency_threshold", 0.8)),
        )
        answer = result.get("answer", "")
        return _make_result(
            answer,
            result.get("chunks", []),
            result.get("time_ms", 0.0),
            estimate_tokens(answer),
            metadata={"active": result.get("active_retrieval")},
        )

    def run_marag(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
            use_marag=True,
            marag_iterations=int(run_cfg.get("marag_iterations", 2)),
        )
        answer = result.get("answer", "")
        return _make_result(
            answer,
            result.get("chunks", []),
            result.get("time_ms", 0.0),
            estimate_tokens(answer),
            metadata={"plan": result.get("plan")},
        )

    def run_madam(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
            use_madam_rag=True,
            madam_followup_rounds=int(run_cfg.get("madam_followup_rounds", 1)),
        )
        answer = result.get("answer", "")
        cited_chunks: List[Dict[str, Any]] = []
        for debater in result.get("debaters", []):
            for resp in debater.get("responses", []):
                cited_chunks.extend(_chunks_from_ids(resp.get("chunk_ids", []), chunk_lookup))
        return _make_result(
            answer,
            cited_chunks,
            result.get("time_ms", 0.0),
            estimate_tokens(answer),
            metadata={"winner": result.get("winner"), "reasoning": result.get("reasoning")},
        )

    def run_routing(question: str) -> Dict[str, Any]:
        result = rag(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            index_map=index_map,
            provider=provider,
            embedding_model=embedding_model,
            llm_model=llm_model,
            k=k,
            threshold=threshold,
            use_routing_agent=True,
        )
        routed_payload = result.get("result", {})
        chunks_used = routed_payload.get("chunks") or routed_payload.get("retrieved_chunks", [])
        time_ms = routed_payload.get("time_ms", result.get("time_ms", 0.0))
        answer = result.get("answer", "")
        metadata = {
            "pipeline": result.get("pipeline"),
            "profile": result.get("profile"),
            "plan": result.get("plan"),
        }
        return _make_result(answer, chunks_used, time_ms, estimate_tokens(answer), metadata=metadata)

    return {
        "vanilla": run_vanilla,
        "self_reflective": run_self_reflective,
        "query_decomposition": run_query_decomposition,
        "chain_of_verification": run_chain_verification,
        "active_retrieval": run_active_retrieval,
        "marag": run_marag,
        "madam_rag": run_madam,
        "routing": run_routing,
    }


def _save_outputs(
    results: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
    output_cfg: Dict[str, Any],
) -> None:
    results_csv = _resolve_path(output_cfg.get("results_csv", "results/experiment_results.csv"))
    results_json = _resolve_path(output_cfg.get("results_json", "results/experiment_results.json"))
    logs_json = _resolve_path(output_cfg.get("logs_json", "results/experiment_logs.json"))

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    logs_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(
        results_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )
    results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logs_json.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run_async(
    questions: List[Dict[str, Any]],
    runners: Dict[str, Callable[[str], Dict[str, Any]]],
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    to_run = config.get("pipelines", {}).get("run", list(runners.keys()))
    to_run = [item for item in to_run if item in runners]
    if not to_run:
        raise ValueError("No valid pipelines requested in config.")

    total_tasks = len(questions) * len(to_run)
    completed = 0
    completed_questions = 0
    question_counts: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    lock = asyncio.Lock()
    concurrency = int(config.get("run", {}).get("concurrency", 4))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    checkpoint_every = int(config.get("run", {}).get("checkpoint_every_questions", 2))
    output_cfg = config.get("output", {})

    async def run_one(question_record: Dict[str, Any], architecture: str) -> None:
        nonlocal completed
        nonlocal completed_questions
        question = question_record["question"]
        runner = runners[architecture]
        async with semaphore:
            logger.info("START | %s | %s", architecture, question)
            try:
                result = await asyncio.to_thread(runner, question)
                error_message = None
            except Exception as exc:
                logger.error("FAILED | %s | %s | error=%s", architecture, question, exc)
                result = {
                    "answer": "",
                    "chunks": [],
                    "time_ms": 0.0,
                    "tokens": 0.0,
                    "metadata": {},
                }
                error_message = str(exc)
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
                logger.warning("EMPTY_ANSWER | skipping semantic metrics | %s | %s", architecture, question)
                sp, sr = 0.0, 0.0
            record = {
                "question": question,
                "question_id": question_record.get("question_id"),
                "question_type": question_record.get("question_type"),
                "architecture": architecture,
                "time_ms": result["time_ms"],
                "tokens": result["tokens"],
                "keyword_precision_k": kp,
                "keyword_recall_k": kr,
                "semantic_precision_k": sp,
                "semantic_recall_k": sr,
                "grounding_score": grounding_score(answer_text or "", chunks_out),
                "error": error_message,
            }
            if architecture == "routing":
                record["routing_pipeline"] = result["metadata"].get("pipeline")
                record["routing_profile"] = (result["metadata"].get("profile") or {}).get("name")

            log_record = {
                "question": question,
                "question_id": question_record.get("question_id"),
                "question_type": question_record.get("question_type"),
                "architecture": architecture,
                "answer": answer_text,
                "chunks": chunks_out,
                "metadata": result.get("metadata", {}),
                "time_ms": result["time_ms"],
                "tokens": result["tokens"],
                "reference_answer": question_record.get("reference_answer"),
                "error": error_message,
            }

            async with lock:
                results.append(record)
                logs.append(log_record)
                completed += 1
                percent = (completed / total_tasks) * 100.0 if total_tasks else 100.0
                logger.success(
                    "DONE | %d/%d | %.1f%% | %s | %s",
                    completed,
                    total_tasks,
                    percent,
                    architecture,
                    question,
                )
                question_counts[question] = question_counts.get(question, 0) + 1
                if question_counts[question] == len(to_run):
                    completed_questions += 1
                    if checkpoint_every > 0 and completed_questions % checkpoint_every == 0:
                        _save_outputs(results, logs, output_cfg)
                        logger.info(
                            "CHECKPOINT | questions=%d/%d | saved",
                            completed_questions,
                            len(questions),
                        )

    tasks = [
        asyncio.create_task(run_one(question_record, architecture))
        for question_record in questions
        for architecture in to_run
    ]
    await asyncio.gather(*tasks)
    return results, logs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG experiments from a YAML config.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _load_yaml(config_path)
    log_path_value = config.get("logging", {}).get("file")
    if log_path_value:
        log_path = _resolve_path(log_path_value)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            logger.addHandler(file_handler)
    questions = _load_questions(config)
    chunks = _load_or_build_chunks(config)
    embeddings, index_map = _load_or_build_embeddings(config, chunks)

    runners = _build_runners(config, chunks, embeddings, index_map)
    results, logs = asyncio.run(
        _run_async(questions, runners, chunks, embeddings, index_map, config)
    )

    output_cfg = config.get("output", {})
    _save_outputs(results, logs, output_cfg)
    logger.info(
        "Saved results to %s",
        _resolve_path(output_cfg.get("results_csv", "results/experiment_results.csv")),
    )
    logger.info(
        "Saved results to %s",
        _resolve_path(output_cfg.get("results_json", "results/experiment_results.json")),
    )
    logger.info(
        "Saved logs to %s",
        _resolve_path(output_cfg.get("logs_json", "results/experiment_logs.json")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
