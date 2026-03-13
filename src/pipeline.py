from __future__ import annotations

from typing import List

from llm import LLM, OpenAILLM
from schemas import (
    ClarificationAnswers,
    ClarifyingQuestion,
    DecisionRequest,
    FinalOutput,
    IterationRecord,
)

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
    current_iteration: int = 0,
    max_iterations: int = 2,
    previous_questions: List[ClarifyingQuestion] | None = None,
    iteration_history: List[IterationRecord] | None = None,
) -> FinalOutput:
    llm = llm or OpenAILLM()
    clarification_answers = clarification_answers or ClarificationAnswers()
    previous_questions = previous_questions or []
    history = list(iteration_history or [])

    orch = Orchestrator(llm=llm)
    critic = CriticAgent(llm=llm)
    synth = Synthesizer(llm=llm)
    q_agent = QuestionerAgent(llm=llm)

    # Initial clarification round before generation.
    if use_questioner and current_iteration == 0 and not clarification_answers.answers:
        q_out = q_agent.run(req, iteration=0)
        if q_out.ask and q_out.questions:
            brief = orch.build_brief(req)
            history.append(
                IterationRecord(
                    iteration=0,
                    status="initial_questioner",
                    summary="Initial clarification requested before generation.",
                    notes=q_out.notes,
                    asked_questions=q_out.questions,
                )
            )

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
            stub.meta.current_iteration = 0
            stub.meta.max_iterations = max_iterations
            stub.meta.iteration_history = history
            stub.meta.clarification_answers = clarification_answers.answers
            return stub

    if clarification_answers.answers:
        brief = orch.build_brief_with_clarification(req, clarification_answers)
    else:
        brief = orch.build_brief(req)

    alt_agent = AlternativesAgent(llm)
    pref_agent = PreferencesAgent(llm)
    unc_agent = UncertaintiesAgent(llm)

    alt_out = alt_agent.run(brief, iteration=current_iteration)
    pref_out = pref_agent.run(brief, iteration=current_iteration)
    unc_out = unc_agent.run(brief, iteration=current_iteration)

    critic_out = critic.review(
        brief=brief,
        alternatives=alt_out.alternatives,
        preferences=pref_out.preferences,
        uncertainties=unc_out.uncertainties,
        iteration=current_iteration,
    )

    history.append(
        IterationRecord(
            iteration=current_iteration,
            status="generation_review",
            summary="Generation agents completed and critic reviewed outputs.",
            notes=critic_out.notes,
            alternatives_count=len(critic_out.alternatives),
            preferences_count=len(critic_out.preferences),
            uncertainties_count=len(critic_out.uncertainties),
            missing_information=critic_out.missing_information,
            received_answers=clarification_answers.answers,
        )
    )

    # Critic-triggered callback to Questioner.
    if use_questioner and critic_out.needs_clarification and current_iteration < max_iterations:
        q_out = q_agent.run_targeted_followup(
            req=req,
            brief=brief,
            missing_information=critic_out.missing_information,
            prior_questions=previous_questions,
            prior_answers=clarification_answers,
            iteration=current_iteration + 1,
            max_questions=4,
        )

        if q_out.ask and q_out.questions:
            history.append(
                IterationRecord(
                    iteration=current_iteration + 1,
                    status="critic_callback_questioner",
                    summary="Critic requested a targeted clarification callback.",
                    notes=critic_out.notes,
                    missing_information=critic_out.missing_information,
                    asked_questions=q_out.questions,
                )
            )

            stub = FinalOutput(
                decision_title=req.title,
                brief=brief,
                alternatives=critic_out.alternatives,
                preferences=critic_out.preferences,
                uncertainties=critic_out.uncertainties,
            )
            stub.meta.used_questioner = True
            stub.meta.pending_clarification = True
            stub.meta.clarifying_questions = q_out.questions
            stub.meta.clarification_answers = clarification_answers.answers
            stub.meta.critic_notes = critic_out.notes
            stub.meta.current_iteration = current_iteration + 1
            stub.meta.max_iterations = max_iterations
            stub.meta.iteration_history = history
            return stub

    final = synth.synthesize(brief=brief, critic_out=critic_out)
    history.append(
        IterationRecord(
            iteration=current_iteration,
            status="finalized",
            summary="Pipeline finalized without needing more clarification.",
            notes=critic_out.notes,
            alternatives_count=len(final.alternatives),
            preferences_count=len(final.preferences),
            uncertainties_count=len(final.uncertainties),
            received_answers=clarification_answers.answers,
        )
    )

    final.meta.used_questioner = use_questioner
    final.meta.pending_clarification = False
    final.meta.clarification_answers = clarification_answers.answers
    final.meta.critic_notes = critic_out.notes
    final.meta.current_iteration = current_iteration
    final.meta.max_iterations = max_iterations
    final.meta.iteration_history = history
    return final