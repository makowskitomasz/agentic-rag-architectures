from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, PositiveInt

VerificationStatus = Literal["verified", "contradicted", "insufficient"]


class VerificationStatement(BaseModel):
    statement_id: PositiveInt
    text: str = Field(..., min_length=5)
    status: VerificationStatus = "insufficient"
    reasoning: str = ""
    supporting_chunks: List[Dict[str, object]] = Field(default_factory=list)
    iterations: int = 0


class VerificationPlan(BaseModel):
    statements: List[str] = Field(default_factory=list, min_length=2)


class ChainVerificationOutput(BaseModel):
    original_answer: str
    refined_answer: str
    statements: List[VerificationStatement]
    iterations: int
    metadata: Dict[str, float] = Field(default_factory=dict)
