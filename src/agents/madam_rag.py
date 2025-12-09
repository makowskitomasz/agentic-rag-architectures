from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.config import get_embedding_config
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedding configuration is required for MADAM-RAG") from exc

try:
    from src.embedder import generate_query_embedding
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Embedder module is required for MADAM-RAG") from exc

try:
    from src.retriever import retrieve
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Retriever module is required for MADAM-RAG") from exc

try:
    from src.reranker import rerank_chunks
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Reranker module is required for MADAM-RAG") from exc

try:
    from src.llm_orchestrator import generate_answer, generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator is required for MADAM-RAG") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("MADAM_RAG")


class DebaterResponse(BaseModel):
    round_id: PositiveInt
    question: str
    answer: str
    chunk_ids: List[str]


class DebaterRecord(BaseModel):
    name: str
    responses: List[DebaterResponse]


class ModeratorFollowUp(BaseModel):
    ask_followup: bool
    question: str = ""


class ModeratorDecision(BaseModel):
    winner: str = Field(..., description="debater_a, debater_b, or merged")
    reasoning: str
    final_answer: str


RetrievalFn = Callable[[str], List[Dict[str, Any]]]
AnswerFn = Callable[[str, List[Dict[str, Any]], str, int], str]
FollowUpFn = Callable[[str, List[DebaterRecord], int], ModeratorFollowUp]
DecisionFn = Callable[[str, List[DebaterRecord]], ModeratorDecision]


def _default_retrieve(
    query: str,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    index_map: Dict[str, int],
    embedding_provider: str,
    embedding_model: str,
    k: int,
    threshold: float,
    use_hybrid: bool,
    lexical_weight: float,
    use_reranker: bool,
    reranker_model: str | None,
    rerank_top_k: Optional[int],
) -> List[Dict[str, Any]]:
    embedding_config = get_embedding_config(embedding_provider)
    query_embedding = generate_query_embedding(
        query,
        provider=embedding_config.provider,
        model=embedding_model or embedding_config.model,
    )
    retrieved = retrieve(
        query_embedding=query_embedding,
        embeddings=embeddings,
        index_map=index_map,
        chunks=chunks,
        k=k,
        threshold=threshold,
        query_text=query,
        use_hybrid=use_hybrid,
        lexical_weight=lexical_weight,
    )
    if use_reranker and retrieved:
        top_k = rerank_top_k if rerank_top_k is not None else k
        retrieved = rerank_chunks(query=query, chunks=retrieved, model_name=reranker_model, top_k=top_k)
    return retrieved


def _default_answer(
    prompt_query: str,
    context_chunks: List[Dict[str, Any]],
    provider: str,
    llm_model: str | None,
    temperature: float,
    debater_name: str,
    round_id: int,
) -> str:
    instruction = (
        f"You are {debater_name}. Answer the user question referencing chunk_id values in brackets, "
        "e.g., [chunk_id=c1]. Keep reasoning concise."
    )
    prompt = f"{instruction}\n\nQuestion:\n{prompt_query}"
    return generate_answer(
        query=prompt,
        context_chunks=context_chunks,
        provider=provider,
        model=llm_model,
        temperature=temperature,
    )


def _default_followup(
    original_query: str,
    debaters: List[DebaterRecord],
    round_id: int,
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> ModeratorFollowUp:
    transcript = "\n\n".join(
        f"{debater.name} (round {resp.round_id}): {resp.answer}"
        for debater in debaters
        for resp in debater.responses[-1:]
    )
    prompt = (
        "You are the moderator in a debate. Decide if a follow-up question is needed "
        "to resolve disagreements between two debaters. If needed, propose a concise question.\n"
        "Respond in JSON with 'ask_followup' (true/false) and optional 'question'.\n\n"
        "Original question:\n{query}\n\n"
        "Latest responses:\n{transcript}\n"
    ).format(query=original_query, transcript=transcript)

    response = generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=ModeratorFollowUp,
    )
    if response.ask_followup and not response.question.strip():
        response.ask_followup = False
    return response


def _default_decision(
    original_query: str,
    debaters: List[DebaterRecord],
    provider: str,
    llm_model: str | None,
    temperature: float,
) -> ModeratorDecision:
    transcript = "\n\n".join(
        f"{debater.name} responses:\n" + "\n".join(f"[Round {resp.round_id}] {resp.answer}" for resp in debater.responses)
        for debater in debaters
    )
    prompt = (
        "You are the moderator concluding the debate. Choose the better answer or merge both. "
        "Return JSON with 'winner' (debater_a, debater_b, merged), 'reasoning', and 'final_answer'. "
        "Final answer must cite chunk IDs as provided by debaters.\n\n"
        "Original question:\n{query}\n\n"
        "Debate transcript:\n{transcript}\n"
    ).format(query=original_query, transcript=transcript)

    return generate_structured_answer(
        query=prompt,
        context_chunks=[],
        provider=provider,
        model=llm_model,
        temperature=temperature,
        response_model=ModeratorDecision,
    )


def run_madam_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[str, int],
    provider: str = "openai",
    llm_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    k: int = 5,
    threshold: float = 0.5,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
    use_reranker: bool = True,
    reranker_model: str | None = None,
    rerank_top_k: Optional[int] = None,
    temperature: float = 1.0,
    followup_rounds: int = 0,
    followup_temperature: float = 1.0,
    debater_names: Sequence[str] = ("Debater A", "Debater B"),
    retrieval_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    answer_fn: Optional[AnswerFn] = None,
    followup_fn: Optional[FollowUpFn] = None,
    decision_fn: Optional[DecisionFn] = None,
) -> Dict[str, Any]:
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if retrieval_fn is None and (embedding_provider is None or embedding_model is None):
        raise ValueError("embedding_provider and embedding_model are required when using default retrieval.")

    start = time.perf_counter()
    logger.info("MADAM_RAG | start | followups=%d", followup_rounds)
    debater_records: List[DebaterRecord] = [DebaterRecord(name=name, responses=[]) for name in debater_names[:2]]

    def _retrieve_wrapper(question: str) -> List[Dict[str, Any]]:
        if retrieval_fn:
            return retrieval_fn(question) or []
        return _default_retrieve(
            question,
            embeddings,
            chunks,
            index_map,
            embedding_provider,
            embedding_model,
            k,
            threshold,
            use_hybrid,
            lexical_weight,
            use_reranker,
            reranker_model,
            rerank_top_k,
        )

    answer_callable = answer_fn or (
        lambda q, ctx, name, rid: _default_answer(q, ctx, provider, llm_model, temperature, name, rid)
    )
    followup_callable = followup_fn or (
        lambda q, records, rid: _default_followup(q, records, rid, provider, llm_model, followup_temperature)
    )
    decision_callable = decision_fn or (
        lambda q, records: _default_decision(q, records, provider, llm_model, temperature)
    )

    def _record_response(record: DebaterRecord, question_text: str, round_id: int) -> None:
        context = _retrieve_wrapper(question_text)
        chunk_ids = [str(chunk.get("chunk_id")) for chunk in context]
        answer_text = answer_callable(question_text, context, record.name, round_id)
        record.responses.append(
            DebaterResponse(round_id=round_id, question=question_text, answer=answer_text, chunk_ids=chunk_ids)
        )

    # Initial round
    for record in debater_records:
        _record_response(record, query, 1)

    # Follow-up rounds
    for round_idx in range(1, followup_rounds + 1):
        followup = followup_callable(query, debater_records, round_idx + 1)
        if not followup.ask_followup:
            break
        question_text = followup.question.strip()
        if not question_text:
            break
        logger.info("MADAM_RAG | moderator follow-up %d: %s", round_idx, question_text)
        for record in debater_records:
            _record_response(record, question_text, round_idx + 1)

    decision = decision_callable(query, debater_records)
    total_time = (time.perf_counter() - start) * 1000
    logger.success("MADAM_RAG | completed | winner=%s time=%.2f ms", decision.winner, total_time)

    return {
        "query": query,
        "answer": decision.final_answer.strip(),
        "winner": decision.winner,
        "reasoning": decision.reasoning,
        "debaters": [record.model_dump() for record in debater_records],
        "followup_rounds": followup_rounds,
        "time_ms": total_time,
        "provider": provider,
        "model": llm_model or "default",
    }
