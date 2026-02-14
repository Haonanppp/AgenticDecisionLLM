from __future__ import annotations

import json
from agenticdq.schemas import DecisionRequest, DecisionBrief
from agenticdq.llm import LLM
from agenticdq.utils import complete_and_validate


class Orchestrator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def build_brief(self, req: DecisionRequest) -> DecisionBrief:
        system = (
            "You are the Orchestrator.\n"
            "Task: Convert (title, narrative) into a structured DecisionBrief.\n"
            "Return JSON ONLY.\n"
            "Rules:\n"
            "- hard_constraints: non-negotiable requirements (budget cap, deadline, cannot/must, etc.)\n"
            "- soft_preferences: negotiable preferences/criteria (prefer/ideally/nice-to-have)\n"
            "- context: structured fields if explicitly mentioned (budget_usd, deadline, location, etc.).\n"
            "- summary: 1-3 sentences concise restatement.\n"
            "OUTPUT_SCHEMA: DecisionBrief\n"
        )

        payload = {"title": req.title, "narrative": req.narrative}
        user = json.dumps(payload, ensure_ascii=False)

        brief = complete_and_validate(self.llm, system=system, user_json=user, model_cls=DecisionBrief, retries=2)
        brief.context.orchestrator_mode = "llm_only"
        return brief
