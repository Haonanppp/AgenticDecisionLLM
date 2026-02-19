from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# --- make "src/" importable on Streamlit Cloud ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import (
    DecisionRequest,
    ClarificationAnswers,
    ClarificationAnswer,
    ClarifyingQuestion,
)
from pipeline import run_mvp
from llm import OpenAILLM


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
          /* Page width */
          .block-container { padding-top: 3.25rem; padding-bottom: 2.0rem; max-width: 1200px; }

          /* Header */
          .adq-header {
            display:flex; align-items:flex-end; justify-content:space-between;
            gap: 0.25rem; margin-bottom: 0.75rem;
          }
          .adq-title { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
          .adq-subtitle { color: rgba(49, 51, 63, 0.7); font-size: 0.98rem; margin-top: 0.2rem; }

          /* Card */
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

          /* Make buttons nicer */
          button[kind="primary"] { border-radius: 12px !important; }
          button { border-radius: 12px !important; }

          /* Sidebar spacing */
          section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }
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


def _answer_widget(q: ClarifyingQuestion, value_key: str):
    """
    Render an input widget based on expected_answer_type.
    Always store values into st.session_state[value_key].
    """
    t = q.expected_answer_type
    opts = q.options or []

    # Keep blank possible for choice/multi_choice
    if t == "choice" and opts:
        return st.selectbox(q.question, options=[""] + opts, key=value_key)
    if t == "multi_choice" and opts:
        return st.multiselect(q.question, options=opts, key=value_key)

    if t == "number":
        # Use text input to allow blank; downstream expects string anyway.
        return st.text_input(q.question, key=value_key, placeholder="Enter a number (e.g., 1500)")
    if t == "date":
        # Use text input to avoid auto-filling a date.
        return st.text_input(q.question, key=value_key, placeholder="YYYY-MM-DD")

    # default free_text
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


def main() -> None:
    st.set_page_config(
        page_title="Agentic Decision LLM",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _load_secrets_into_env()

    # --- Header ---
    st.markdown(
        """
        <div class="adq-header">
          <div>
            <div class="adq-title">Agentic Decision LLM</div>
            <div class="adq-subtitle">Structured decision support from a title + narrative → brief, alternatives, preferences, uncertainties.</div>
          </div>
          <div class="adq-badge">v0.2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Settings")
        _require_api_key_ui()

        default_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        model = st.text_input("OpenAI model", value=default_model)

        use_questioner = st.checkbox(
            "Use Questioner (clarification)",
            value=False,
            help="When enabled, the system may ask clarifying questions before generating outputs.",
        )

        st.markdown("---")
        st.markdown("### Quick start")
        ex_name = st.selectbox("Load an example", options=list(_examples().keys()), index=1)

        with st.expander("Advanced", expanded=False):
            show_raw = st.checkbox("Show raw JSON tab", value=True)
            exclude_none = st.checkbox("Hide null fields in JSON", value=True)
            st.caption("Tip: Hiding null fields makes output cleaner for users.")

        st.markdown("---")
        st.caption("Deployment tip: keep your API key in Streamlit Secrets, not in code.")

    # --- Session state init ---
    if "example_name" not in st.session_state:
        st.session_state.example_name = ex_name
        st.session_state.title = _examples()[ex_name]["title"]
        st.session_state.narrative = _examples()[ex_name]["narrative"]

    # Update inputs only when user changes the example selection
    if ex_name != st.session_state.example_name:
        st.session_state.example_name = ex_name
        st.session_state.title = _examples()[ex_name]["title"]
        st.session_state.narrative = _examples()[ex_name]["narrative"]

        # Clear prior run states when switching examples
        st.session_state.last_output = None
        st.session_state.last_run_meta = None
        st.session_state.pending_sig = None
        st.session_state.pending_questions = []

    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "last_run_meta" not in st.session_state:
        st.session_state.last_run_meta = None
    if "pending_sig" not in st.session_state:
        st.session_state.pending_sig = None
    if "pending_questions" not in st.session_state:
        st.session_state.pending_questions = []

    # --- Input card ---
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

    # Persist latest edits to session
    st.session_state.title = title
    st.session_state.narrative = narrative

    # --- Run ---
    if run_btn:
        if not title.strip() or not narrative.strip():
            st.warning("Please provide both decision title and narrative.")
            st.stop()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Please configure it in Secrets or environment variables.")
            st.stop()

        # Clear previous question inputs if user re-runs
        st.session_state.pending_questions = []
        st.session_state.pending_sig = None
        # Also clear stored answers for previous questions
        for k in list(st.session_state.keys()):
            if str(k).startswith("q_ans_"):
                del st.session_state[k]

        try:
            llm = _get_llm(model.strip() or None)
            req = DecisionRequest(title=title.strip(), narrative=narrative.strip())

            with st.spinner("Running pipeline..."):
                out = run_mvp(req, llm=llm, use_questioner=use_questioner)

            st.session_state.last_output = out
            st.session_state.last_run_meta = {
                "model": model.strip() or default_model,
                "time": _fmt_time_utc(),
                "n_alts": len(out.alternatives),
                "n_prefs": len(out.preferences),
                "n_uncs": len(out.uncertainties),
                "use_questioner": use_questioner,
            }

            # If pending clarification, store signature + questions for UI
            if out.meta.pending_clarification and out.meta.clarifying_questions:
                st.session_state.pending_sig = _req_signature(req.title, req.narrative)
                st.session_state.pending_questions = out.meta.clarifying_questions

            st.success("Done!")

        except Exception as e:
            st.error(f"Error: {e}")

    out = st.session_state.last_output
    meta = st.session_state.last_run_meta

    if out is None:
        st.info("Load an example or enter your own decision, then click **Run**.")
        return

    # --- Clarification UI (if pending) ---
    if out.meta.pending_clarification and st.session_state.pending_questions:
        st.markdown('<div class="adq-card">', unsafe_allow_html=True)
        st.markdown("#### Clarification (Questioner)")
        st.info("Answer the questions below, then click **Run with answers** to continue.")

        # Safety: ensure inputs did not change after questions were generated
        current_sig = _req_signature(st.session_state.title, st.session_state.narrative)
        if st.session_state.pending_sig and current_sig != st.session_state.pending_sig:
            st.warning("Your title/narrative changed after questions were generated. Please click **Run** again.")
        else:
            with st.form("clar_form", clear_on_submit=False):
                for q in st.session_state.pending_questions:
                    key = f"q_ans_{q.id}"
                    _answer_widget(q, key)
                    if q.rationale:
                        st.caption(f"Why this matters: {q.rationale}")

                run_with_answers = st.form_submit_button("Run with answers", type="primary", use_container_width=True)

            if run_with_answers:
                llm = _get_llm((meta["model"] if meta else default_model).strip() or None)
                req = DecisionRequest(title=st.session_state.title.strip(), narrative=st.session_state.narrative.strip())
                clar = _build_clarification_answers(st.session_state.pending_questions)

                with st.spinner("Running pipeline with clarification answers..."):
                    out2 = run_mvp(req, llm=llm, use_questioner=True, clarification_answers=clar)

                st.session_state.last_output = out2
                st.session_state.last_run_meta = {
                    "model": meta["model"] if meta else default_model,
                    "time": _fmt_time_utc(),
                    "n_alts": len(out2.alternatives),
                    "n_prefs": len(out2.preferences),
                    "n_uncs": len(out2.uncertainties),
                    "use_questioner": True,
                }
                st.session_state.pending_questions = []
                st.session_state.pending_sig = None
                st.success("Done!")
                out = out2
                meta = st.session_state.last_run_meta

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Results ---
    st.markdown('<div class="adq-card">', unsafe_allow_html=True)
    st.markdown("#### Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alternatives", meta["n_alts"] if meta else len(out.alternatives))
    m2.metric("Preferences", meta["n_prefs"] if meta else len(out.preferences))
    m3.metric("Uncertainties", meta["n_uncs"] if meta else len(out.uncertainties))
    m4.metric("Model", meta["model"] if meta else "—")

    if meta:
        st.caption(f"Last run: {meta['time']} • Questioner: {'ON' if meta.get('use_questioner') else 'OFF'}")

    # Tabs
    base_tabs = ["Overview", "Brief", "Clarification", "Alternatives", "Preferences", "Uncertainties"]
    if show_raw:
        base_tabs.append("Raw JSON")
    tabs = st.tabs(base_tabs)

    # Overview
    with tabs[0]:
        st.markdown("**Summary**")
        st.write(out.brief.summary)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hard constraints**")
            if out.brief.hard_constraints:
                for x in out.brief.hard_constraints:
                    st.markdown(f"- {x}")
            else:
                st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)

        with c2:
            st.markdown("**Soft preferences**")
            if out.brief.soft_preferences:
                for x in out.brief.soft_preferences:
                    st.markdown(f"- {x}")
            else:
                st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)

    # Brief
    with tabs[1]:
        st.subheader("Decision Brief")
        st.json(out.brief.model_dump(exclude_none=True))

    # Clarification tab
    with tabs[2]:
        st.subheader("Questioner (Clarification)")
        if out.meta.used_questioner or out.meta.pending_clarification:
            if out.meta.clarifying_questions:
                st.markdown("**Questions**")
                for q in out.meta.clarifying_questions:
                    st.markdown(f"- **{q.id}** ({q.category}) {q.question}")
            else:
                st.markdown('<span class="adq-muted">No questions were asked.</span>', unsafe_allow_html=True)

            if out.meta.clarification_answers:
                st.markdown("**Your answers**")
                for a in out.meta.clarification_answers:
                    st.markdown(f"- **{a.question_id}**: {a.answer}")
            elif out.meta.pending_clarification:
                st.markdown('<span class="adq-muted">Waiting for answers.</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="adq-muted">No answers were provided.</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="adq-muted">Questioner was disabled for this run.</span>', unsafe_allow_html=True)

    # Alternatives
    with tabs[3]:
        st.subheader("Alternatives")
        if not out.alternatives:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for i, a in enumerate(out.alternatives, 1):
            with st.expander(f"{i}. {a.text}", expanded=(i <= 2)):
                if a.rationale:
                    st.write(a.rationale)
                st.caption(f"Source: {a.provenance.agent} • iteration {a.provenance.iteration}")

    # Preferences
    with tabs[4]:
        st.subheader("Preferences")
        if not out.preferences:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for i, p in enumerate(out.preferences, 1):
            with st.expander(f"{i}. {p.text}", expanded=(i <= 2)):
                if p.rationale:
                    st.write(p.rationale)
                st.caption(f"Source: {p.provenance.agent} • iteration {p.provenance.iteration}")

    # Uncertainties
    with tabs[5]:
        st.subheader("Uncertainties")
        if not out.uncertainties:
            st.markdown('<span class="adq-muted">None</span>', unsafe_allow_html=True)
        for i, u in enumerate(out.uncertainties, 1):
            with st.expander(f"{i}. {u.text}", expanded=(i <= 2)):
                if u.rationale:
                    st.write(u.rationale)
                st.caption(f"Source: {u.provenance.agent} • iteration {u.provenance.iteration}")

    # Raw JSON
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
