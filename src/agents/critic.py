from __future__ import annotations

import json
from typing import List

from llm import LLM
from schemas import CriticOutput, DecisionBrief, Item
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
            "You are the Critic Agent.\n"
            "You must clean, repair, and assess whether additional clarification is needed.\n"
            "Return JSON ONLY.\n"
            "Goals:\n"
            "- Remove duplicates, vague items, infeasible items, and low-quality items.\n"
            "- Reclassify miscategorized items into the correct bucket.\n"
            "- Rewrite items for clarity when needed, without inventing user facts.\n"
            "- Judge whether the current information is sufficient for a detailed and decision-useful answer.\n"
            "Clarification policy:\n"
            "- Set needs_clarification=true only when missing user information materially limits specificity, feasibility, tradeoff analysis, or uncertainty framing.\n"
            "- missing_information must list 1-4 concrete information gaps.\n"
            "- If outputs are already sufficiently specific and useful, set needs_clarification=false and return an empty missing_information list.\n"
            "Bucket rules:\n"
            "- Alternatives must be actionable choices or plans.\n"
            "- Preferences must be evaluation criteria or value dimensions.\n"
            "- Uncertainties must be unknowns that could change which alternative is best.\n"
            "Constraints:\n"
            "- Keep 4-8 alternatives, 5-10 preferences, 5-10 uncertainties when possible.\n"
            "- Do NOT invent facts that the user did not provide.\n"
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

        out = complete_and_validate(
            self.llm,
            system=system,
            user_json=user,
            model_cls=CriticOutput,
            retries=2,
        )

        for item in out.alternatives:
            item.type = "alternative"
            item.provenance.agent = "critic"
            item.provenance.iteration = iteration

        for item in out.preferences:
            item.type = "preference"
            item.provenance.agent = "critic"
            item.provenance.iteration = iteration

        for item in out.uncertainties:
            item.type = "uncertainty"
            item.provenance.agent = "critic"
            item.provenance.iteration = iteration

        if len(out.missing_information) > 4:
            out.missing_information = out.missing_information[:4]

        if not out.missing_information:
            out.needs_clarification = False

        return out