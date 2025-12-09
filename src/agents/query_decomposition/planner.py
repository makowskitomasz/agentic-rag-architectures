from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

try:
    from src.utils import get_logger as _get_logger
except ImportError:
    _get_logger = None

try:
    from src.llm_orchestrator import generate_structured_answer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLM orchestrator module is required for query planner") from exc

try:
    from src.models.query_decomposition_models import QueryDecompositionPlan, QueryPlanStep
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Query decomposition models are required for query planner") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("QUERY_PLANNER")

HeuristicFn = Callable[[str, int], Sequence[str]]


def _split_on_delimiters(query: str, max_subqueries: int) -> List[str]:
    parts: List[str] = []
    for delimiter in ["?", ".", ";"]:
        if delimiter in query:
            parts = [chunk.strip() for chunk in query.split(delimiter) if chunk.strip()]
            break
    if not parts:
        return []
    return parts[:max_subqueries]


def _split_on_conjunctions(query: str, max_subqueries: int) -> List[str]:
    lowered = query.lower()
    for conjunction in [" and ", " oraz ", " oraz też ", " oraz także ", " as well as "]:
        if conjunction in lowered:
            fragments = [frag.strip().capitalize() for frag in query.split(conjunction) if frag.strip()]
            return fragments[:max_subqueries]
    return []


class QueryPlanner:
    def __init__(
        self,
        provider: str = "openai",
        llm_model: str | None = None,
        heuristics: Optional[List[HeuristicFn]] = None,
        min_subqueries: int = 2,
    ) -> None:
        self.provider = provider
        self.llm_model = llm_model
        self.min_subqueries = max(2, min_subqueries)
        self.heuristics: List[HeuristicFn] = heuristics[:] if heuristics else [_split_on_delimiters, _split_on_conjunctions]

    def add_heuristic(self, heuristic: HeuristicFn) -> None:
        self.heuristics.append(heuristic)

    def plan(
        self,
        query: str,
        max_subqueries: int = 3,
        use_llm: bool = False,
        temperature: float = 1.0,
    ) -> QueryDecompositionPlan:
        candidate_subqueries = self._apply_heuristics(query, max_subqueries)
        if len(candidate_subqueries) < self.min_subqueries and use_llm:
            logger.info("QUERY_PLANNER | invoking LLM planner for query: %s", query)
            candidate_subqueries = self._llm_plan(query, max_subqueries, temperature=temperature)
        if len(candidate_subqueries) < self.min_subqueries:
            candidate_subqueries = self._fallback_split(query, max_subqueries)

        steps = [
            QueryPlanStep(step_id=idx + 1, question=subquery, rationale="Heuristic decomposition")
            for idx, subquery in enumerate(candidate_subqueries)
        ]
        plan = QueryDecompositionPlan(original_query=query, steps=steps)
        logger.info("QUERY_PLANNER | plan generated with %d sub-queries", len(steps))
        return plan

    def _apply_heuristics(self, query: str, max_subqueries: int) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for heuristic in self.heuristics:
            try:
                suggestions = heuristic(query, max_subqueries)
            except Exception as exc:  # pragma: no cover - heuristic failure should not crash pipeline
                logger.warning("QUERY_PLANNER | heuristic failed: %s", exc)
                continue
            for suggestion in suggestions:
                cleaned = suggestion.strip()
                if not cleaned or cleaned.lower() in seen:
                    continue
                seen.add(cleaned.lower())
                ordered.append(cleaned)
                if len(ordered) >= max_subqueries:
                    return ordered
        return ordered

    def _llm_plan(self, query: str, max_subqueries: int, temperature: float) -> List[str]:
        from pydantic import BaseModel, Field

        class LLMPlan(BaseModel):
            steps: List[str] = Field(default_factory=list, min_length=self.min_subqueries)

        prompt = (
            "You decompose complex research questions into focused sub-questions.\n"
            "Return a JSON list named 'steps' containing between {min_sub} and {max_sub} sub-questions.\n"
            "Each sub-question must address a distinct aspect of the original query.\n\n"
            "Original query:\n{query}"
        ).format(min_sub=self.min_subqueries, max_sub=max_subqueries, query=query)

        response = generate_structured_answer(
            query=prompt,
            context_chunks=[],
            provider=self.provider,
            model=self.llm_model,
            temperature=temperature,
            response_model=LLMPlan,
        )
        return response.steps[:max_subqueries]

    def _fallback_split(self, query: str, max_subqueries: int) -> List[str]:
        midpoint = len(query) // 2
        first = query[:midpoint].strip()
        second = query[midpoint:].strip()
        results = [segment for segment in [first, second] if segment]
        while len(results) < self.min_subqueries:
            results.append(query.strip())
        return results[:max_subqueries]
