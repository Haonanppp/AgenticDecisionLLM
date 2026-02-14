from __future__ import annotations

from agenticdq.llm import LLM
from agenticdq.schemas import DecisionBrief, AlternativesOutput
from agenticdq.utils import complete_and_validate


class AlternativesAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, brief: DecisionBrief, iteration: int = 0) -> AlternativesOutput:
        system = (
            "You are the Alternatives Agent.\n"
            "Generate actionable, mutually distinguishable alternatives.\n"
            "Rules:\n"
            "- Each item must be an action/plan the user can choose.\n"
            "- Avoid preferences/criteria; avoid uncertainties.\n"
            "- Return JSON only.\n"
            "OUTPUT_SCHEMA: AlternativesOutput\n"
        )

        user = (
            f"TITLE: {brief.title}\n"
            f"BRIEF: {brief.summary}\n"
            f"HARD_CONSTRAINTS: {brief.hard_constraints}\n"
            f"SOFT_PREFERENCES: {brief.soft_preferences}\n"
            "Return 5-8 alternatives.\n"
        )

        out = complete_and_validate(
            llm=self.llm,
            system=system,
            user_json=user,
            model_cls=AlternativesOutput,
            retries=2,
        )

        # harden provenance/type
        for item in out.alternatives:
            item.type = "alternative"
            item.provenance.agent = "alternatives"
            item.provenance.iteration = iteration

        return out
