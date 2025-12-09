from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class CritiqueModel(BaseModel):
    missing_context: List[str] = Field(default_factory=list, description="List of missing context points identified in the initial answer.")
    conflicts: List[str] = Field(default_factory=list, description="List of conflicts identified in the initial answer.")
    logic_issues: List[str] = Field(default_factory=list, description="List of logical issues identified in the initial answer.")
    hallucinations: List[str] = Field(default_factory=list, description="List of hallucinations identified in the initial answer.")
    precision_warnings: List[str] = Field(default_factory=list, description="List of precision warnings identified in the initial answer.")
    language_problems: List[str] = Field(default_factory=list, description="List of language problems identified in the initial answer.")
    reasoning_gaps: List[str] = Field(default_factory=list, description="List of reasoning gaps identified in the initial answer.")


class SelfReflectiveRagOutput(BaseModel):
    query: str
    initial_answer: str
    critique: CritiqueModel
    refined_answer: str
    retrieved_chunks: List[Dict[str, object]]
    timings: Dict[str, float]
    provider: str
    model: str
