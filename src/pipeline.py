from __future__ import annotations

from typing import Callable

from schemas import DecisionRequest, FinalOutput, ClarificationAnswers
from llm import LLM, OpenAILLM

from agents.orchestrator import Orchestrator
from agents.alternatives import AlternativesAgent
from agents.preferences import PreferencesAgent
from agents.uncertainties import UncertaintiesAgent
from agents.critic import CriticAgent
from agents.synthesizer import Synthesizer
from agents.questioner import QuestionerAgent

ProgressCallback = Callable[[str, int], None]  # (stage_label, percent_0_100)

def run_mvp(
    req: DecisionRequest,
    llm: LLM | None = None,
    use_questioner: bool = False,
    clarification_answers: ClarificationAnswers | None = None,
    progress: ProgressCallback | None = None,
) -> FinalOutput:
    def tick(label: str, pct: int) -> None:
        if progress is not None:
            progress(label, max(0, min(100, int(pct))))

    llm = llm or OpenAILLM()

    tick("Initializing...", 3)

    orch = Orchestrator(llm=llm)
    critic = CriticAgent(llm=llm)
    synth = Synthesizer(llm=llm)

    # --- Phase 1: ask questions (return early) ---
    if use_questioner and clarification_answers is None:
        tick("Generating clarification questions...", 10)
        q_agent = QuestionerAgent(llm=llm)
        q_out = q_agent.run(req, iteration=0)

        # If model decides no need to ask, continue normally
        if q_out.ask and q_out.questions:
            tick("Summarizing decision brief...", 22)
            brief = orch.build_brief(req)

            stub = FinalOutput(
                decision_title=req.title,
                brief=brief,
                alternatives=[],
                preferences=[],
                uncertainties=[],
            )
            stub.meta.used_questioner = True
            stub.meta.pending_clarification = True
            stub.meta.clarifying_questions = q_out.questions
            tick("Waiting for answers...", 28)
            return stub

        tick("No clarification needed...", 18)

    # --- Phase 2: build brief (with answers if provided) ---
    if use_questioner and clarification_answers is not None:
        tick("Integrating answers into brief...", 22)
        brief = orch.build_brief_with_clarification(req, clarification_answers)
    else:
        tick("Building decision brief...", 22)
        brief = orch.build_brief(req)

    alt_agent = AlternativesAgent(llm)
    pref_agent = PreferencesAgent(llm)
    unc_agent = UncertaintiesAgent(llm)

    tick("Generating alternatives...", 40)
    alt_out = alt_agent.run(brief, iteration=0)

    tick("Generating preferences...", 58)
    pref_out = pref_agent.run(brief, iteration=0)

    tick("Generating uncertainties...", 72)
    unc_out = unc_agent.run(brief, iteration=0)

    tick("Critic review...", 85)
    critic_out = critic.review(
        brief=brief,
        alternatives=alt_out.alternatives,
        preferences=pref_out.preferences,
        uncertainties=unc_out.uncertainties,
        iteration=0,
    )

    tick("Synthesizing final output...", 95)
    final = synth.synthesize(brief=brief, critic_out=critic_out)

    # Fill meta flags
    final.meta.used_questioner = use_questioner
    final.meta.pending_clarification = False
    if clarification_answers is not None:
        final.meta.clarification_answers = clarification_answers.answers

    tick("Done", 100)
    return final