from __future__ import annotations

import json
from schemas import DecisionRequest, DecisionBrief, ClarificationAnswers
from llm import LLM
from utils import complete_and_validate


class Orchestrator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def build_brief(self, req: DecisionRequest) -> DecisionBrief:
        system = (
            "You are the Orchestrator.\n"
            "Task: Convert (title, narrative) into a structured DecisionBrief.\n"
            "Return JSON ONLY.\n"
            "Rules:\n"
            "- Keep title exactly the same as the input title.\n"
            "- summary: 1-3 concise sentences.\n"
            "- hard_constraints: only non-negotiables explicitly stated or logically required.\n"
            "- soft_preferences: negotiable preferences/criteria and tradeoffs.\n"
            "- Do NOT invent facts not provided by the user.\n"
            "OUTPUT_SCHEMA: DecisionBrief\n"
        )

        payload = {"title": req.title, "narrative": req.narrative}
        user = json.dumps(payload, ensure_ascii=False)

        return complete_and_validate(
            llm=self.llm,
            system=system,
            user_json=user,
            model_cls=DecisionBrief,
            retries=2,
        )

    def build_brief_with_clarification(self, req: DecisionRequest, clar: ClarificationAnswers) -> DecisionBrief:
        system = (
            "You are the Orchestrator.\n"
            "Task: Convert (title, narrative, clarification answers) into a structured DecisionBrief.\n"
            "Return JSON ONLY.\n"
            "Rules:\n"
            "- Keep title exactly the same as the input title.\n"
            "- summary: 1-3 concise sentences.\n"
            "- hard_constraints: non-negotiable requirements extracted from answers and narrative.\n"
            "- soft_preferences: negotiable preferences/criteria extracted from answers and narrative.\n"
            "- Do NOT invent facts not provided by the user.\n"
            "OUTPUT_SCHEMA: DecisionBrief\n"
        )

        payload = {
            "title": req.title,
            "narrative": req.narrative,
            "clarification_answers": [a.model_dump() for a in clar.answers],
        }
        user = json.dumps(payload, ensure_ascii=False)

        return complete_and_validate(
            llm=self.llm,
            system=system,
            user_json=user,
            model_cls=DecisionBrief,
            retries=2,
        )