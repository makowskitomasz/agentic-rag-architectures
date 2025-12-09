from __future__ import annotations

import logging
from typing import List, Tuple

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.llm_orchestrator import generate_answer
except ImportError as exc:
    raise RuntimeError("LLM orchestrator module is required for query aggregation") from exc

try:
    from src.models.query_decomposition_models import QueryDecompositionPlan, SubQueryAnswer
except ImportError as exc:
    raise RuntimeError("Query decomposition models are required for aggregation") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("QUERY_AGGREGATOR")


class AnswerAggregator:
    def __init__(
        self,
        provider: str = "openai",
        llm_model: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.provider = provider
        self.llm_model = llm_model
        self.temperature = temperature

    def aggregate(
        self,
        original_query: str,
        plan: QueryDecompositionPlan,
        answers: List[SubQueryAnswer],
    ) -> Tuple[str, str]:
        sub_answer_digest = "\n\n".join(
            f"[Step {ans.step_id}] Sub-query: {ans.sub_query}\nAnswer:\n{ans.answer}"
            for ans in answers
        )
        prompt = (
            "You are an aggregator agent. Using the sub-answers below, synthesize a single coherent response "
            "to the original user question. Ensure logical consistency and cite which sub-steps informed critical claims.\n\n"
            "Original question:\n{query}\n\n"
            "Sub-answers:\n{digest}\n"
        ).format(query=original_query, digest=sub_answer_digest)

        logger.info("QUERY_AGGREGATOR | start | answers=%d", len(answers))
        content = generate_answer(
            query=prompt,
            context_chunks=[],
            provider=self.provider,
            model=self.llm_model,
            temperature=self.temperature,
        )
        notes = f"Aggregated from {len(answers)} sub-answers using model {self.llm_model or 'default'}."
        logger.success("QUERY_AGGREGATOR | aggregation completed")
        return content.strip(), notes
