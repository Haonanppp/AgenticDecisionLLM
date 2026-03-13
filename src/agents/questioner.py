from __future__ import annotations

import json
from typing import List

from llm import LLM
from schemas import (
    ClarificationAnswers,
    ClarifyingQuestion,
    DecisionBrief,
    DecisionRequest,
    QuestionerOutput,
)
from utils import complete_and_validate


class QuestionerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, req: DecisionRequest, iteration: int = 0) -> QuestionerOutput:
        system = (
            "You are the Questioner Agent.\n"
            "Goal: Ask 3-8 clarifying questions that reduce ambiguity for downstream decision analysis.\n"
            "Rules:\n"
            "- Ask questions only. Do NOT assume facts or fill in answers.\n"
            "- Prioritize: hard constraints, then preferences/tradeoffs, then key uncertainties.\n"
            "- Each question must be actionable for building a DecisionBrief.\n"
            "- Avoid duplicates. Keep it concise.\n"
            "Return JSON ONLY.\n"
            "OUTPUT_SCHEMA: QuestionerOutput\n"
        )

        payload = {"title": req.title, "narrative": req.narrative, "iteration": iteration}
        user = json.dumps(payload, ensure_ascii=False)

        out = complete_and_validate(
            self.llm,
            system=system,
            user_json=user,
            model_cls=QuestionerOutput,
            retries=2,
        )

        if out.ask and len(out.questions) > 8:
            out.questions = out.questions[:8]

        return out

    def run_targeted_followup(
        self,
        req: DecisionRequest,
        brief: DecisionBrief,
        missing_information: List[str],
        prior_questions: List[ClarifyingQuestion] | None = None,
        prior_answers: ClarificationAnswers | None = None,
        iteration: int = 1,
        max_questions: int = 4,
    ) -> QuestionerOutput:
        system = (
            "You are the Questioner Agent.\n"
            "Goal: Generate a small set of highly targeted follow-up clarification questions\n"
            "based on the Critic's identified information gaps.\n"
            "Rules:\n"
            "- Ask only questions that directly resolve the listed missing information.\n"
            "- Do NOT repeat or paraphrase prior questions unless necessary.\n"
            "- Prefer 1-4 focused questions rather than broad brainstorming.\n"
            "- Make each question answerable and decision-relevant.\n"
            "- Do NOT ask for trivia or unnecessary personal detail.\n"
            "Return JSON ONLY.\n"
            "OUTPUT_SCHEMA: QuestionerOutput\n"
        )

        payload = {
            "title": req.title,
            "narrative": req.narrative,
            "brief": brief.model_dump(),
            "missing_information": missing_information,
            "prior_questions": [q.model_dump() for q in (prior_questions or [])],
            "prior_answers": [a.model_dump() for a in (prior_answers.answers if prior_answers else [])],
            "iteration": iteration,
        }
        user = json.dumps(payload, ensure_ascii=False)

        out = complete_and_validate(
            self.llm,
            system=system,
            user_json=user,
            model_cls=QuestionerOutput,
            retries=2,
        )

        out.ask = bool(out.questions)
        if len(out.questions) > max_questions:
            out.questions = out.questions[:max_questions]

        for idx, q in enumerate(out.questions, start=1):
            q.id = f"followup_{iteration}_{idx}"

        return out