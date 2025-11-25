from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class CritiqueModel(BaseModel):
    missing_context: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    logic_issues: List[str] = Field(default_factory=list)
    hallucinations: List[str] = Field(default_factory=list)
    precision_warnings: List[str] = Field(default_factory=list)
    language_problems: List[str] = Field(default_factory=list)
    reasoning_gaps: List[str] = Field(default_factory=list)


class SelfReflectiveRagOutput(BaseModel):
    query: str
    initial_answer: str
    critique: CritiqueModel
    refined_answer: str
    retrieved_chunks: List[Dict[str, object]]
    timings: Dict[str, float]
    provider: str
    model: str
