from __future__ import annotations

from llm import LLM
from schemas import DecisionBrief, UncertaintiesOutput
from utils import complete_and_validate


class UncertaintiesAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, brief: DecisionBrief, iteration: int = 0) -> UncertaintiesOutput:
        system = (
            "You are the Uncertainties Agent.\n"
            "Identify key unknowns that could change which alternative is best.\n"
            "Rules:\n"
            "- Each item must be an uncertainty/unknown (NOT a preference, NOT an alternative).\n"
            "- Focus on uncertainties that affect tradeoffs (cost, performance, timing, constraints).\n"
            "- Return JSON only.\n"
            "OUTPUT_SCHEMA: UncertaintiesOutput\n"
        )

        user = (
            f"TITLE: {brief.title}\n"
            f"BRIEF: {brief.summary}\n"
            f"HARD_CONSTRAINTS: {brief.hard_constraints}\n"
            f"SOFT_PREFERENCES: {brief.soft_preferences}\n"
            "Return 5-10 uncertainties.\n"
        )

        out = complete_and_validate(
            llm=self.llm,
            system=system,
            user_json=user,
            model_cls=UncertaintiesOutput,
            retries=2,
        )

        for item in out.uncertainties:
            item.type = "uncertainty"
            item.provenance.agent = "uncertainties"
            item.provenance.iteration = iteration

        return out
