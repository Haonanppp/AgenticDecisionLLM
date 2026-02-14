from __future__ import annotations

import json
from agenticdq.llm import LLM
from agenticdq.schemas import DecisionBrief, CriticOutput, FinalOutput
from agenticdq.utils import complete_and_validate


class Synthesizer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def synthesize(self, brief: DecisionBrief, critic_out: CriticOutput) -> FinalOutput:
        system = (
            "You are the Synthesizer.\n"
            "Task: Produce the final structured output for downstream UI.\n"
            "Return JSON ONLY.\n"
            "Rules:\n"
            "- Use the given brief and cleaned lists.\n"
            "- Do NOT invent facts.\n"
            "- meta can include a short synthesis_summary and critic_notes.\n"
            "OUTPUT_SCHEMA: FinalOutput\n"
        )

        payload = {
            "brief": brief.model_dump(),
            "critic_out": critic_out.model_dump(),
        }
        user = json.dumps(payload, ensure_ascii=False)

        final = complete_and_validate(self.llm, system=system, user_json=user, model_cls=FinalOutput, retries=2)
        return final
