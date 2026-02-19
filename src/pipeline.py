from __future__ import annotations

from schemas import DecisionRequest, FinalOutput, ClarificationAnswers
from llm import LLM, OpenAILLM

from agents.orchestrator import Orchestrator
from agents.alternatives import AlternativesAgent
from agents.preferences import PreferencesAgent
from agents.uncertainties import UncertaintiesAgent
from agents.critic import CriticAgent
from agents.synthesizer import Synthesizer
from agents.questioner import QuestionerAgent


def run_mvp(
    req: DecisionRequest,
    llm: LLM | None = None,
    use_questioner: bool = False,
    clarification_answers: ClarificationAnswers | None = None,
) -> FinalOutput:
    llm = llm or OpenAILLM()

    orch = Orchestrator(llm=llm)
    critic = CriticAgent(llm=llm)
    synth = Synthesizer(llm=llm)

    # --- Phase 1: ask questions (return early) ---
    if use_questioner and clarification_answers is None:
        q_agent = QuestionerAgent(llm=llm)
        q_out = q_agent.run(req, iteration=0)

        # If model decides no need to ask, continue normally
        if q_out.ask and q_out.questions:
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
            return stub

    # --- Phase 2: build brief (with answers if provided) ---
    if use_questioner and clarification_answers is not None:
        brief = orch.build_brief_with_clarification(req, clarification_answers)
    else:
        brief = orch.build_brief(req)

    alt_agent = AlternativesAgent(llm)
    pref_agent = PreferencesAgent(llm)
    unc_agent = UncertaintiesAgent(llm)

    alt_out = alt_agent.run(brief, iteration=0)
    pref_out = pref_agent.run(brief, iteration=0)
    unc_out = unc_agent.run(brief, iteration=0)

    critic_out = critic.review(
        brief=brief,
        alternatives=alt_out.alternatives,
        preferences=pref_out.preferences,
        uncertainties=unc_out.uncertainties,
        iteration=0,
    )

    final = synth.synthesize(brief=brief, critic_out=critic_out)

    # Fill meta flags
    final.meta.used_questioner = use_questioner
    final.meta.pending_clarification = False
    if clarification_answers is not None:
        final.meta.clarification_answers = clarification_answers.answers

    return final