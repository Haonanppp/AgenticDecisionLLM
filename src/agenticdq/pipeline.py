from __future__ import annotations

from agenticdq.schemas import DecisionRequest, FinalOutput
from agenticdq.llm import LLM, OpenAILLM

from agenticdq.agents.orchestrator import Orchestrator
from agenticdq.agents.alternatives import AlternativesAgent
from agenticdq.agents.preferences import PreferencesAgent
from agenticdq.agents.uncertainties import UncertaintiesAgent
from agenticdq.agents.critic import CriticAgent
from agenticdq.agents.synthesizer import Synthesizer


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
