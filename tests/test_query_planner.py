from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.agents.query_decomposition.planner import QueryPlanner


def test_planner_produces_minimum_subqueries() -> None:
    planner = QueryPlanner()
    plan = planner.plan("Explain supply chain resilience and energy security strategies for Poland?", max_subqueries=3)
    assert len(plan.steps) >= 2
    assert plan.steps[0].step_id == 1
    assert isinstance(plan.steps[0].question, str) and plan.steps[0].question


def test_custom_heuristic_is_applied_first() -> None:
    def custom_heuristic(query: str, _: int) -> Sequence[str]:
        return [f"{query} part A", f"{query} part B", f"{query} part C"]

    planner = QueryPlanner(heuristics=[custom_heuristic])
    plan = planner.plan("Any question", max_subqueries=2)
    assert len(plan.steps) == 2
    assert plan.steps[0].question.endswith("part A")
    assert plan.steps[1].question.endswith("part B")
