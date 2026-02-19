from __future__ import annotations

import json
from llm import LLM
from schemas import DecisionRequest, QuestionerOutput
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

        payload = {"title": req.title, "narrative": req.narrative}
        user = json.dumps(payload, ensure_ascii=False)

        out = complete_and_validate(self.llm, system=system, user_json=user, model_cls=QuestionerOutput, retries=2)

        if out.ask and len(out.questions) > 8:
            out.questions = out.questions[:8]

        return out