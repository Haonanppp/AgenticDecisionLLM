from __future__ import annotations

import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

# --- make "src/" importable on Streamlit Cloud ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import (
    ClarificationAnswer,
    ClarificationAnswers,
    ClarifyingQuestion,
    DecisionRequest,
    IterationRecord,
)
from pipeline import run_mvp
from llm import OpenAILLM


APP_VERSION = "v0.3"


def _load_secrets_into_env() -> None:
    """
    On Streamlit Community Cloud, secrets are provided via UI.
    We read st.secrets and set env vars for OpenAI SDK compatibility.
    """
    try:
        if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if "OPENAI_MODEL" in st.secrets and not os.getenv("OPENAI_MODEL"):
            os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
    except Exception:
        pass


@st.cache_resource
def _get_llm(model: str | None):
    return OpenAILLM(model=model)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 3.25rem; padding-bottom: 2.0rem; max-width: 1200px; }

          .adq-header {
            display:flex; align-items:flex-end; justify-content:space-between;
            gap: 0.25rem; margin-bottom: 0.75rem;
          }
          .adq-title { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
          .adq-subtitle { color: rgba(49, 51, 63, 0.7); font-size: 0.98rem; margin-top: 0.2rem; }

          .adq-card {
            border: 1px solid rgba(49,51,63,0.12);
            border-radius: 16px;
            padding: 16px 16px;
            background: rgba(255,255,255,0.65);
            box-shadow: 0 1px 0 rgba(49,51,63,0.03);
          }
          .adq-card + .adq-card { margin-top: 12px; }

          .adq-badge {
            display:inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            border: 1px solid rgba(49,51,63,0.15);
            background: rgba(49,51,63,0.03);
          }

          .adq-muted { color: rgba(49, 51, 63, 0.65); }
          button[kind="primary"] { border-radius: 12px !important; }
          button { border-radius: 12px !important; }
          section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

          div[data-testid="stProgress"] > div > div > div {
            height: 10px;
            border-radius: 999px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _examples() -> dict[str, dict[str, str]]:
    return {
        "Blank (start fresh)": {"title": "", "narrative": ""},
        "Choose a laptop": {
            "title": "Choose a laptop",
            "narrative": "Budget $1500, prefer lightweight, mainly for coding and ML.",
        },
        "Choose a job offer": {
            "title": "Choose between two job offers",
            "narrative": "Offer A: higher salary but long commute. Offer B: lower salary but remote-friendly and better growth. I value learning and work-life balance.",
        },
        "Choose a research direction": {
            "title": "Choose a research direction",
            "narrative": "I’m deciding between optimization theory, agentic AI evaluation, and applied NLP. I want publishable work within 6–9 months and strong PhD fit.",
        },
        "Choose a travel destination": {
            "title": "Choose a warm winter trip",
            "narrative": "I want to travel in winter to a warm destination. Budget is $4000 USD for a 5-day trip. It must be international (not within the US).",
        },
    }


def _require_api_key_ui() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "Missing `OPENAI_API_KEY`. "
            "If running on Streamlit Cloud, set it in **App → Settings → Secrets**. "
            "If running locally, put it in `.env` or your shell env."
        )


def _fmt_time_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _req_signature(title: str, narrative: str) -> str:
    return f"{title.strip()}\n---\n{narrative.strip()}"


def _reset_workflow_state(clear_output: bool = True) -> None:
    st.session_state.pending_sig = None
    st.session_state.pending_questions = []
    st.session_state.all_questions = []
    st.session_state.all_answers = []
    st.session_state.iteration_history = []
    st.session_state.current_iteration = 0
    st.session_state.show_clar_panel = True
    st.session_state.clar_run_requested = False
    st.session_state.clar_run_payload = None

    for key in list(st.session_state.keys()):
        if str(key).startswith("q_ans_"):
            del st.session_state[key]

    if clear_output:
        st.session_state.last_output = None
        st.session_state.last_run_meta = None


def _merge_questions(
    existing: list[ClarifyingQuestion],
    new_questions: list[ClarifyingQuestion],
) -> list[ClarifyingQuestion]:
    merged: dict[str, ClarifyingQuestion] = {q.id: q for q in existing}
    for q in new_questions:
        merged[q.id] = q
    return list(merged.values())


def _merge_answers(
    existing: list[ClarificationAnswer],
    new_answers: list[ClarificationAnswer],
) -> list[ClarificationAnswer]:
    merged: dict[str, ClarificationAnswer] = {a.question_id: a for a in existing}
    for a in new_answers:
        merged[a.question_id] = a
    return list(merged.values())


def _answer_widget(q: ClarifyingQuestion, value_key: str) -> Any:
    """
    Render an input widget based on expected_answer_type.
    Always store values into st.session_state[value_key].
    """
    answer_type = q.expected_answer_type
    options = q.options or []

    if answer_type == "choice" and options:
        return st.selectbox(q.question, options=[""] + options, key=value_key)
    if answer_type == "multi_choice" and options:
        return st.multiselect(q.question, options=options, key=value_key)
    if answer_type == "number":
        return st.text_input(q.question, key=value_key, placeholder="Enter a number")
    if answer_type == "date":
        return st.text_input(q.question, key=value_key, placeholder="YYYY-MM-DD")
    return st.text_input(q.question, key=value_key)


def _build_clarification_answers(questions: list[ClarifyingQuestion]) -> ClarificationAnswers:
    answers: list[ClarificationAnswer] = []
    for q in questions:
        key = f"q_ans_{q.id}"
        raw = st.session_state.get(key, "")
        if isinstance(raw, list):
            raw = ", ".join([str(x) for x in raw if str(x).strip()])
        raw_str = str(raw).strip()
        if raw_str:
            answers.append(ClarificationAnswer(question_id=q.id, answer=raw_str))
    return ClarificationAnswers(answers=answers)


def _run_pipeline_with_status(
    header_label: str,
    *,
    req: DecisionRequest,
    llm,
    use_questioner: bool,
    clarification_answers: ClarificationAnswers | None = None,
    current_iteration: int = 0,
    max_iterations: int = 2,
    previous_questions: list[ClarifyingQuestion] | None = None,
    iteration_history: list[IterationRecord] | None = None,
):
    progress_bar = st.progress(0)
    status_fn = getattr(st, "status", None)
    status_box = status_fn(header_label, expanded=True) if status_fn is not None else None
    fallback = st.empty() if status_box is None else None

    def set_stage(text: str, pct: int) -> None:
        pct_int = max(0, min(100, int(pct)))
        progress_bar.progress(pct_int)
        if status_box is not None:
            status_box.update(label=text, state="running", expanded=True)
            status_box.write(text)
        else:
            fallback.info(text)

    kwargs = {
        "req": req,
        "llm": llm,
        "use_questioner": use_questioner,
        "clarification_answers": clarification_answers,
        "current_iteration": current_iteration,
        "max_iterations": max_iterations,
        "previous_questions": previous_questions,
        "iteration_history": iteration_history,
    }

    try:
        sig = inspect.signature(run_mvp)
        if "progress" in sig.parameters:
            kwargs["progress"] = lambda stage, pct: set_stage(stage, pct)
        else:
            set_stage("Running pipeline...", 15)

        out = run_mvp(**kwargs)

        if status_box is not None:
            if getattr(out, "meta", None) is not None and out.meta.pending_clarification:
                status_box.update(
                    label="Clarification needed — please answer the questions below.",
                    state="complete",
                    expanded=True,
                )
            else:
                status_box.update(label="Done", state="complete", expanded=False)
        progress_bar.progress(100)
        return out
    except Exception as exc:
        if status_box is not None:
            status_box.update(label=f"Error: {exc}", state="error", expanded=True)
        raise


def _render_clarification_tab() -> None:
    st.subheader("Questioner (Clarification)")

    all_questions: list[ClarifyingQuestion] = st.session_state.all_questions
    all_answers: list[ClarificationAnswer] = st.session_state.all_answers
    answer_map = {a.question_id: a.answer for a in all_answers}

    if not all_questions and not st.session_state.pending_questions:
        st.markdown('<span class="adq-muted">No questions were asked.</span>', unsafe_allow_html=True)
        return

    if all_questions:
        st.markdown("**Questions and answers**")
        for q in all_questions:
            st.markdown(f"- **{q.id}** ({q.category}) {q.question}")
            if q.rationale:
                st.caption(f"Why this matters: {q.rationale}")
            answer = answer_map.get(q.id)
            if answer:
                st.markdown(f"  - **Answer:** {answer}")
            else:
                st.markdown("  - **Answer:** _pending_")

    if st.session_state.pending_questions:
        st.markdown("---")
        st.info("There are still pending clarification questions to answer before the next pipeline run.")


def _render_iteration_history_tab() -> None:
    st.subheader("Iteration History")
    history: list[IterationRecord] = st.session_state.iteration_history

    if not history:
        st.markdown('<span class="adq-muted">No iteration history available.</span>', unsafe_allow_html=True)
        return

    for idx, rec in enumerate(history, start=1):
        title = f"{idx}. iteration={rec.iteration} • status={rec.status}"
        with st.expander(title, expanded=(idx == len(history))):
            if rec.summary:
                st.write(rec.summary)
            if rec.notes:
                st.markdown("**Notes**")
                for note in rec.notes:
                    st.markdown(f"- {note}")
            st.markdown(
                f"**Counts:** alternatives={rec.alternatives_count}, "
                f"preferences={rec.preferences_count}, uncertainties={rec.uncertainties_count}"
            )
            if rec.missing_information:
                st.markdown("**Missing information**")
                for item in rec.missing_information:
                    st.markdown(f"- {item}")
            if rec.asked_questions:
                st.markdown("**Asked questions**")
                for q in rec.asked_questions:
                    st.markdown(f"- **{q.id}** ({q.category}) {q.question}")
            if rec.received_answers:
                st.markdown("**Received answers**")
                for a in rec.received_answers:
                    st.markdown(f"- **{a.question_id}**: {a.answer}")


def main() -> None:
    st.set_page_config(
        page_title="Agentic Decision LLM",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _load_secrets_into_env()

    st.markdown(
        f"""
        <div class="adq-header">
          <div>
            <div class="adq-title">Agentic Decision LLM</div>
            <div class="adq-subtitle">Structured decision support from a title + narrative → brief, alternatives, preferences, uncertainties, critic review, and iterative clarification.</div>
          </div>
          <div class="adq-badge">{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Settings")
        _require_api_key_ui()

        default_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        preset_models = ["gpt-5.4", "gpt-5-mini", "Custom"]
        default_model_clean = default_model.strip() or "gpt-5-mini"
        default_model_choice = default_model_clean if default_model_clean in {"gpt-5.4", "gpt-5-mini"} else "Custom"

        model_choice = st.selectbox(
            "OpenAI model",
            options=preset_models,
            index=preset_models.index(default_model_choice),
            help="Choose a preset model or select Custom to enter your own model name.",
        )
        custom_model_value = default_model_clean if default_model_choice == "Custom" else ""
        model = (
            st.text_input(
                "Custom model name",
                value=custom_model_value,
                placeholder="e.g., gpt-5.4",
            ).strip()
            if model_choice == "Custom"
            else model_choice
        )

        use_questioner = st.checkbox(
            "Use Questioner (clarification)",
            value=True,
            help="Allow the pipeline to ask initial and critic-triggered follow-up questions.",
        )
        max_iterations = st.slider(
            "Max clarification iterations",
            min_value=0,
            max_value=4,
            value=2,
            help="Maximum number of critic-triggered clarification callbacks.",
        )

        st.markdown("---")
        st.markdown("### Quick start")
        ex_name = st.selectbox("Load an example", options=list(_examples().keys()), index=1)

        with st.expander("Advanced", expanded=False):
            show_raw = st.checkbox("Show raw JSON tab", value=True)
            exclude_none = st.checkbox("Hide null fields in JSON", value=True)
            keep_clar_open = st.checkbox(
                "Keep clarification panel expanded when pending",
                value=True,
            )

        st.markdown("---")
        st.caption("Deployment tip: keep your API key in Streamlit Secrets, not in code.")

    if "example_name" not in st.session_state:
        st.session_state.example_name = ex_name
        st.session_state.title = _examples()[ex_name]["title"]
        st.session_state.narrative = _examples()[ex_name]["narrative"]

    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "last_run_meta" not in st.session_state:
        st.session_state.last_run_meta = None
    if "pending_sig" not in st.session_state:
        st.session_state.pending_sig = None
    if "pending_questions" not in st.session_state:
        st.session_state.pending_questions = []
    if "all_questions" not in st.session_state:
        st.session_state.all_questions = []
    if "all_answers" not in st.session_state:
        st.session_state.all_answers = []
    if "iteration_history" not in st.session_state:
        st.session_state.iteration_history = []
    if "current_iteration" not in st.session_state:
        st.session_state.current_iteration = 0
    if "show_clar_panel" not in st.session_state:
        st.session_state.show_clar_panel = True
    if "clar_run_requested" not in st.session_state:
        st.session_state.clar_run_requested = False
    if "clar_run_payload" not in st.session_state:
        st.session_state.clar_run_payload = None

    if ex_name != st.session_state.example_name:
        st.session_state.example_name = ex_name
        st.session_state.title = _examples()[ex_name]["title"]
        st.session_state.narrative = _examples()[ex_name]["narrative"]
        _reset_workflow_state(clear_output=True)

    st.markdown('<div class="adq-card">', unsafe_allow_html=True)
    st.markdown("#### Input")

    with st.form("adq_form", clear_on_submit=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            title = st.text_input(
                "Decision title",
                value=st.session_state.title,
                placeholder="e.g., Choose a laptop",
            )
        with col2:
            st.markdown(" ")
            st.markdown(" ")
            run_btn = st.form_submit_button("Run", type="primary", use_container_width=True)

        narrative = st.text_area(
            "Decision narrative",
            value=st.session_state.narrative,
            height=140,
            placeholder="Provide context, constraints, preferences, timeline, and anything you already know...",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.title = title
    st.session_state.narrative = narrative

    if run_btn:
        if not title.strip() or not narrative.strip():
            st.warning("Please provide both decision title and narrative.")
            st.stop()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Please configure it in Secrets or environment variables.")
            st.stop()

        _reset_workflow_state(clear_output=True)
        st.session_state.show_clar_panel = keep_clar_open

        try:
            llm = _get_llm(model.strip() or None)
            req = DecisionRequest(title=title.strip(), narrative=narrative.strip())
            out = _run_pipeline_with_status(
                "Running pipeline...",
                req=req,
                llm=llm,
                use_questioner=use_questioner,
                clarification_answers=ClarificationAnswers(),
                current_iteration=0,
                max_iterations=max_iterations,
                previous_questions=[],
                iteration_history=[],
            )

            st.session_state.last_output = out
            st.session_state.current_iteration = out.meta.current_iteration
            st.session_state.iteration_history = list(out.meta.iteration_history)
            st.session_state.all_answers = list(out.meta.clarification_answers)
            st.session_state.pending_questions = list(out.meta.clarifying_questions) if out.meta.pending_clarification else []
            st.session_state.all_questions = _merge_questions(
                st.session_state.all_questions,
                list(out.meta.clarifying_questions),
            )

            if out.meta.pending_clarification and out.meta.clarifying_questions:
                st.session_state.pending_sig = _req_signature(req.title, req.narrative)
                st.info("Clarification required. Please answer the questions below.")
            else:
                st.success("Done!")

            st.session_state.last_run_meta = {
                "model": model.strip() or default_model,
                "time": _fmt_time_utc(),
                "n_alts": len(out.alternatives),
                "n_prefs": len(out.preferences),
                "n_uncs": len(out.uncertainties),
                "use_questioner": use_questioner,
                "current_iteration": out.meta.current_iteration,
                "max_iterations": max_iterations,
                "pending": out.meta.pending_clarification,
            }

        except Exception as exc:
            st.error(f"Error: {exc}")

    if st.session_state.clar_run_requested and st.session_state.clar_run_payload:
        payload = st.session_state.clar_run_payload
        try:
            llm = _get_llm((payload.get("model") or os.getenv("OPENAI_MODEL", "gpt-5-mini")).strip() or None)
            req2 = DecisionRequest(title=payload["title"], narrative=payload["narrative"])
            all_answers = ClarificationAnswers(answers=payload["all_answers"])

            out2 = _run_pipeline_with_status(
                "Running pipeline with clarification answers...",
                req=req2,
                llm=llm,
                use_questioner=True,
                clarification_answers=all_answers,
                current_iteration=payload["current_iteration"],
                max_iterations=payload["max_iterations"],
                previous_questions=payload["previous_questions"],
                iteration_history=payload["iteration_history"],
            )

            st.session_state.last_output = out2
            st.session_state.current_iteration = out2.meta.current_iteration
            st.session_state.iteration_history = list(out2.meta.iteration_history)
            st.session_state.all_answers = _merge_answers(
                st.session_state.all_answers,
                list(out2.meta.clarification_answers),
            )
            st.session_state.pending_questions = list(out2.meta.clarifying_questions) if out2.meta.pending_clarification else []
            st.session_state.all_questions = _merge_questions(
                payload["previous_questions"],
                list(out2.meta.clarifying_questions),
            )

            if out2.meta.pending_clarification and out2.meta.clarifying_questions:
                st.session_state.pending_sig = _req_signature(req2.title, req2.narrative)
                st.session_state.show_clar_panel = keep_clar_open
                st.info("More clarification is needed. Please answer the next round of questions.")
            else:
                st.session_state.pending_sig = None
                st.session_state.show_clar_panel = False
                st.success("Done!")

            st.session_state.last_run_meta = {
                "model": payload.get("model") or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                "time": _fmt_time_utc(),
                "n_alts": len(out2.alternatives),
                "n_prefs": len(out2.preferences),
                "n_uncs": len(out2.uncertainties),
                "use_questioner": True,
                "current_iteration": out2.meta.current_iteration,
                "max_iterations": payload["max_iterations"],
                "pending": out2.meta.pending_clarification,
            }
        except Exception as exc:
            st.error(f"Error: {exc}")
        finally:
            st.session_state.clar_run_requested = False
            st.session_state.clar_run_payload = None

    out = st.session_state.last_output
    meta = st.session_state.last_run_meta

    if out is None:
        st.info("Load an example or enter your own decision, then click **Run**.")
        return

    has_pending = bool(st.session_state.pending_questions)
    show_panel = st.session_state.show_clar_panel

    if has_pending:
        st.markdown('<div class="adq-card">', unsafe_allow_html=True)

        h1, h2 = st.columns([0.78, 0.22])
        with h1:
            st.markdown("#### Clarification (Questioner)")
        with h2:
            btn_label = "Collapse" if show_panel else "Expand"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_clar_panel = not st.session_state.show_clar_panel
                show_panel = st.session_state.show_clar_panel

        if show_panel:
            current_sig = _req_signature(st.session_state.title, st.session_state.narrative)
            if st.session_state.pending_sig and current_sig != st.session_state.pending_sig:
                st.warning("Your title or narrative changed after questions were generated. Please click **Run** again.")
            else:
                with st.form("clar_form", clear_on_submit=False):
                    for q in st.session_state.pending_questions:
                        key = f"q_ans_{q.id}"
                        _answer_widget(q, key)
                        if q.rationale:
                            st.caption(f"Why this matters: {q.rationale}")

                    run_with_answers = st.form_submit_button(
                        "Run with answers",
                        type="primary",
                        use_container_width=True,
                    )

                if run_with_answers:
                    new_answers = _build_clarification_answers(st.session_state.pending_questions)
                    if not new_answers.answers:
                        st.warning("Please answer at least one clarification question before continuing.")
                    else:
                        merged_answers = _merge_answers(st.session_state.all_answers, new_answers.answers)
                        previous_questions = _merge_questions(
                            st.session_state.all_questions,
                            st.session_state.pending_questions,
                        )

                        st.session_state.all_answers = merged_answers
                        st.session_state.all_questions = previous_questions
                        st.session_state.clar_run_payload = {
                            "title": st.session_state.title.strip(),
                            "narrative": st.session_state.narrative.strip(),
                            "model": meta["model"] if meta else (model.strip() or default_model),
                            "all_answers": merged_answers,
                            "previous_questions": previous_questions,
                            "current_iteration": out.meta.current_iteration,
                            "max_iterations": meta["max_iterations"] if meta else max_iterations,
                            "iteration_history": st.session_state.iteration_history,
                        }
                        st.session_state.clar_run_requested = True
                        st.session_state.pending_questions = []
                        st.session_state.show_clar_panel = False
                        st.rerun()
        else:
            st.caption("Clarification is pending. Click **Expand** to answer questions.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="adq-card">', unsafe_allow_html=True)
    st.markdown("#### Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alternatives", meta["n_alts"] if meta else len(out.alternatives))
    m2.metric("Preferences", meta["n_prefs"] if meta else len(out.preferences))
    m3.metric("Uncertainties", meta["n_uncs"] if meta else len(out.uncertainties))
    m4.metric("Iteration", f"{out.meta.current_iteration}/{out.meta.max_iterations}")

    if meta:
        st.caption(
            f"Last run: {meta['time']} • Model: {meta['model']} • "
            f"Questioner: {'ON' if meta.get('use_questioner') else 'OFF'} • "
            f"Pending clarification: {'YES' if meta.get('pending') else 'NO'}"
        )

    tab_names = [
        "Overview",
        "Brief",
        "Clarification",
        "Alternatives",
        "Preferences",
        "Uncertainties",
        "Critic",
        "Iterations",
    ]
    if show_raw:
        tab_names.append("Raw JSON")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if out.meta.synthesis_summary:
            st.markdown("**Synthesis summary**")
            st.write(out.meta.synthesis_summary)
            st.markdown("---")

        st.markdown("**Summary**")
        st.write(out.brief.summary)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hard constraints**")
            if out.brief.hard_constraints:
                for item in out.brief.hard_constraints:
                    st.markdown(f"- {item}")
            else:
                st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)

        with c2:
            st.markdown("**Soft preferences**")
            if out.brief.soft_preferences:
                for item in out.brief.soft_preferences:
                    st.markdown(f"- {item}")
            else:
                st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("Decision Brief")
        st.json(out.brief.model_dump(exclude_none=True))

    with tabs[2]:
        _render_clarification_tab()

    with tabs[3]:
        st.subheader("Alternatives")
        if not out.alternatives:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for idx, item in enumerate(out.alternatives, start=1):
            with st.expander(f"{idx}. {item.text}", expanded=(idx <= 2)):
                if item.rationale:
                    st.write(item.rationale)
                st.caption(f"Source: {item.provenance.agent} • iteration {item.provenance.iteration}")

    with tabs[4]:
        st.subheader("Preferences")
        if not out.preferences:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for idx, item in enumerate(out.preferences, start=1):
            with st.expander(f"{idx}. {item.text}", expanded=(idx <= 2)):
                if item.rationale:
                    st.write(item.rationale)
                st.caption(f"Source: {item.provenance.agent} • iteration {item.provenance.iteration}")

    with tabs[5]:
        st.subheader("Uncertainties")
        if not out.uncertainties:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for idx, item in enumerate(out.uncertainties, start=1):
            with st.expander(f"{idx}. {item.text}", expanded=(idx <= 2)):
                if item.rationale:
                    st.write(item.rationale)
                st.caption(f"Source: {item.provenance.agent} • iteration {item.provenance.iteration}")

    with tabs[6]:
        st.subheader("Critic")
        notes = out.meta.critic_notes or []
        if notes:
            st.markdown("**Critic notes**")
            for note in notes:
                st.markdown(f"- {note}")
        else:
            st.markdown('<span class="adq-muted">No critic notes.</span>', unsafe_allow_html=True)

    with tabs[7]:
        _render_iteration_history_tab()

    if show_raw:
        with tabs[-1]:
            dump_kwargs = {"exclude_none": True} if exclude_none else {}
            data = out.model_dump(**dump_kwargs)
            st.json(data)
            st.download_button(
                "Download JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="agenticdq_output.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
