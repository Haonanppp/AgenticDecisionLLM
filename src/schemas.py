from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class DecisionRequest(BaseModel):
    title: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)


class DecisionBrief(BaseModel):
    title: str
    summary: str
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)

class ClarifyingQuestion(BaseModel):
    id: str = Field(..., min_length=1)  # e.g. "q1"
    category: Literal["hard_constraint", "soft_preference", "uncertainty", "context"]
    question: str = Field(..., min_length=5)
    expected_answer_type: Literal["free_text", "number", "date", "choice", "multi_choice"] = "free_text"
    options: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None


class QuestionerOutput(BaseModel):
    ask: bool = True
    questions: List[ClarifyingQuestion] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ClarificationAnswer(BaseModel):
    question_id: str
    answer: str = Field(..., min_length=1)


class ClarificationAnswers(BaseModel):
    answers: List[ClarificationAnswer] = Field(default_factory=list)


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
    model_config = ConfigDict(extra="forbid")

    mvp: bool = True
    synthesis_summary: Optional[str] = None
    critic_notes: List[str] = Field(default_factory=list)

    used_questioner: bool = False
    pending_clarification: bool = False
    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)
    clarification_answers: List[ClarificationAnswer] = Field(default_factory=list)


class FinalOutput(BaseModel):
    decision_title: str
    brief: DecisionBrief
    alternatives: List[Item]
    preferences: List[Item] = Field(default_factory=list)
    uncertainties: List[Item] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)
