from __future__ import annotations

import json
from typing import List
from llm import LLM
from schemas import DecisionBrief, Item, CriticOutput
from utils import complete_and_validate


class CriticAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def review(
        self,
        brief: DecisionBrief,
        alternatives: List[Item],
        preferences: List[Item],
        uncertainties: List[Item],
        iteration: int = 0,
    ) -> CriticOutput:
        system = (
            "You are the Critic.\n"
            "You will clean and correct the outputs.\n"
            "Return JSON ONLY.\n"
            "Goals:\n"
            "- Remove duplicates and low-quality items.\n"
            "- Reclassify miscategorized items into correct type.\n"
            "- Ensure alternatives are actionable choices (not criteria, not questions).\n"
            "- Ensure preferences are evaluation criteria (not actions).\n"
            "- Ensure uncertainties are unknowns that could change the best choice.\n"
            "Constraints:\n"
            "- Keep 4-8 alternatives, 5-10 preferences, 5-10 uncertainties (if possible).\n"
            "- Do NOT invent user facts. You may rewrite for clarity.\n"
            "OUTPUT_SCHEMA: CriticOutput\n"
        )

        payload = {
            "brief": brief.model_dump(),
            "alternatives": [x.model_dump() for x in alternatives],
            "preferences": [x.model_dump() for x in preferences],
            "uncertainties": [x.model_dump() for x in uncertainties],
            "iteration": iteration,
        }
        user = json.dumps(payload, ensure_ascii=False)

        out = complete_and_validate(self.llm, system=system, user_json=user, model_cls=CriticOutput, retries=2)

        # Optional hardening: force types to match buckets (avoid model mistakes)
        for it in out.alternatives:
            it.type = "alternative"
        for it in out.preferences:
            it.type = "preference"
        for it in out.uncertainties:
            it.type = "uncertainty"

        return out