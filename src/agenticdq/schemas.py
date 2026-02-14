from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class DecisionRequest(BaseModel):
    title: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)


class DecisionContext(BaseModel):
    """
    IMPORTANT for Structured Outputs:
    This must be a closed object (no arbitrary keys).
    """
    model_config = ConfigDict(extra="forbid")

    budget_usd: Optional[int] = None
    budget_mentioned: Optional[bool] = None
    max_weight_kg: Optional[float] = None
    os: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = None

    orchestrator_mode: Optional[str] = None


class DecisionBrief(BaseModel):
    title: str
    summary: str
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)
    context: DecisionContext = Field(default_factory=DecisionContext)


class Provenance(BaseModel):
    agent: str
    iteration: int = 0


class Item(BaseModel):
    type: Literal["alternative", "preference", "uncertainty"]
    text: str = Field(..., min_length=3)
    rationale: Optional[str] = None
    provenance: Provenance


class AlternativesOutput(BaseModel):
    alternatives: List[Item]


class PreferencesOutput(BaseModel):
    preferences: List[Item]


class UncertaintiesOutput(BaseModel):
    uncertainties: List[Item]


class CriticOutput(BaseModel):
    alternatives: List[Item]
    preferences: List[Item]
    uncertainties: List[Item]
    notes: List[str] = Field(default_factory=list)


class Meta(BaseModel):
    """
    Also must be closed for Structured Outputs.
    """
    model_config = ConfigDict(extra="forbid")

    mvp: bool = True
    synthesis_summary: Optional[str] = None
    critic_notes: List[str] = Field(default_factory=list)


class FinalOutput(BaseModel):
    decision_title: str
    brief: DecisionBrief
    alternatives: List[Item]
    preferences: List[Item] = Field(default_factory=list)
    uncertainties: List[Item] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)
