from __future__ import annotations

from agenticdq.llm import LLM
from agenticdq.schemas import DecisionBrief, PreferencesOutput
from agenticdq.utils import complete_and_validate


class PreferencesAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, brief: DecisionBrief, iteration: int = 0) -> PreferencesOutput:
        system = (
            "You are the Preferences Agent.\n"
            "Extract evaluation criteria / preferences to compare alternatives.\n"
            "Rules:\n"
            "- Each item must be a criterion or preference (NOT an alternative).\n"
            "- Make criteria comparable/measurable when possible.\n"
            "- Avoid uncertainties and questions.\n"
            "- Return JSON only.\n"
            "OUTPUT_SCHEMA: PreferencesOutput\n"
        )

        user = (
            f"TITLE: {brief.title}\n"
            f"BRIEF: {brief.summary}\n"
            f"HARD_CONSTRAINTS: {brief.hard_constraints}\n"
            f"SOFT_PREFERENCES: {brief.soft_preferences}\n"
            "Return 5-10 preferences/criteria.\n"
        )

        out = complete_and_validate(
            llm=self.llm,
            system=system,
            user_json=user,
            model_cls=PreferencesOutput,
            retries=2,
        )

        for item in out.preferences:
            item.type = "preference"
            item.provenance.agent = "preferences"
            item.provenance.iteration = iteration

        return out
