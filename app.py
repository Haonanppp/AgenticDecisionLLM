from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import streamlit as st

# --- make "src/" importable on Streamlit Cloud ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import DecisionRequest
from pipeline import run_mvp
from llm import OpenAILLM


def _load_secrets_into_env():
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
        # st.secrets may not exist locally if you didn't create .streamlit/secrets.toml
        pass


@st.cache_resource
def _get_llm(model: str | None):
    return OpenAILLM(model=model)


def main():
    st.set_page_config(page_title="AgenticDQ", layout="wide")
    _load_secrets_into_env()

    st.title("AgenticDQ MVP")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
        run_btn = st.button("Run", type="primary")

    title = st.text_input("Decision title", value="Choose a laptop")
    narrative = st.text_area(
        "Decision narrative",
        value="Budget $1500, prefer lightweight, mainly for coding and ML",
        height=140,
    )

    if run_btn:
        if not title.strip() or not narrative.strip():
            st.warning("Please provide both title and narrative.")
            st.stop()

        try:
            llm = _get_llm(model.strip() or None)
            req = DecisionRequest(title=title.strip(), narrative=narrative.strip())

            with st.spinner("Running pipeline..."):
                out = run_mvp(req, llm=llm)

            st.success("Done!")

            tabs = st.tabs(["Brief", "Alternatives", "Preferences", "Uncertainties", "Raw JSON"])

            with tabs[0]:
                st.subheader("Decision Brief")
                st.json(out.brief.model_dump())

            with tabs[1]:
                for i, a in enumerate(out.alternatives, 1):
                    st.markdown(f"**{i}. {a.text}**")
                    if a.rationale:
                        st.caption(a.rationale)

            with tabs[2]:
                for i, p in enumerate(out.preferences, 1):
                    st.markdown(f"**{i}. {p.text}**")
                    if p.rationale:
                        st.caption(p.rationale)

            with tabs[3]:
                for i, u in enumerate(out.uncertainties, 1):
                    st.markdown(f"**{i}. {u.text}**")
                    if u.rationale:
                        st.caption(u.rationale)

            with tabs[4]:
                data = out.model_dump()
                st.json(data)
                st.download_button(
                    "Download JSON",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name="agenticdq_output.json",
                    mime="application/json",
                )

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
