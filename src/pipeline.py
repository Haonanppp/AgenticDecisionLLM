from __future__ import annotations

from schemas import DecisionRequest, FinalOutput
from llm import LLM, OpenAILLM

from agents.orchestrator import Orchestrator
from agents.alternatives import AlternativesAgent
from agents.preferences import PreferencesAgent
from agents.uncertainties import UncertaintiesAgent
from agents.critic import CriticAgent
from agents.synthesizer import Synthesizer


def run_mvp(req: DecisionRequest, llm: LLM | None = None) -> FinalOutput:
    llm = llm or OpenAILLM()

    orch = Orchestrator(llm=llm)          # LLM-only
    critic = CriticAgent(llm=llm)         # LLM-only
    synth = Synthesizer(llm=llm)          # LLM-only

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
    return final
