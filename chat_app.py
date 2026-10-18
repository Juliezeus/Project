"""Local employee-facing chat demo. Run with the project .venv interpreter."""
import os
import logging
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import truststore

try:
    from hr_rag.corpus import load_corpus
    from hr_rag.retrieval import PineconeHybrid
    from hr_rag.workflow import build_workflow, chat_model
except ModuleNotFoundError as exc:
    if exc.name != "hr_rag":
        raise
    # GitHub's file picker can upload package files at repository root.
    from corpus import load_corpus
    from retrieval import PineconeHybrid
    from workflow import build_workflow, chat_model

st.set_page_config(page_title="Harbourlight HR assistant", page_icon=":material/policy:")
st.title("Harbourlight HR assistant")
st.caption("Ask about leave, benefits, pay and reward. Answers use fictional company policies for this project.")

load_dotenv(Path(__file__).parent / ".env", encoding="utf-8-sig")
# Community Cloud supplies these values through its Secrets settings.
# Only app configuration is copied; credentials never enter browser state.
try:
    for name in (
        "PINECONE_API_KEY", "PINECONE_DENSE_INDEX", "PINECONE_SPARSE_INDEX",
        "PINECONE_NAMESPACE", "LLM_BACKEND", "CHAT_MODEL", "NEBIUS_API_KEY",
        "NEBIUS_BASE_URL", "LLM_TIMEOUT_SECONDS", "OPENAI_API_KEY",
    ):
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass
truststore.inject_into_ssl()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)

with st.sidebar:
    st.header("Your context")
    st.caption("Demo inputs help select policy rules. They do not grant access to employee records.")
    bu = st.selectbox("Home business unit", ["Not specified", "CORP", "TECH", "OPS"], key="bu")
    known_fte = st.checkbox("I know my FTE", key="known_fte")
    fte = st.number_input("Full-time equivalent", min_value=0.01, max_value=1.0, value=1.0, step=0.1, disabled=not known_fte, key="fte")
    live = st.toggle("Generate answers", value=True, help="Uses Pinecone search and the model configured in .env. Turn off for local evidence preview.")
    if st.button("New conversation", icon=":material/refresh:"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.caption("Policy source: HDS-HR-001 · version 1.1 · effective 1 October 2026")


@st.cache_resource(max_entries=2)
def workflow(live_mode, model_name, backend, source_hash):
    docs = load_corpus()
    return build_workflow(docs, PineconeHybrid() if live_mode else None, chat_model() if live_mode else None)


def render(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for section, content in message.get("sources", {}).items():
            with st.expander(f"Policy source · {section}"):
                st.markdown(content)
        if "seconds" in message:
            st.caption(f"Response time: {message['seconds']:.1f} seconds")


for message in st.session_state.messages:
    render(message)

if not st.session_state.messages:
    st.info("Try: What learning budget applies to a part-time Technology employee?")

if question := st.chat_input("Ask an HR policy question", submit_mode="disable", max_chars=3000):
    previous = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render(user_message)
    start = time.perf_counter()
    stage = "configuration"
    try:
        required = ["PINECONE_API_KEY", "CHAT_MODEL", "LLM_BACKEND"]
        backend = os.getenv("LLM_BACKEND", "").lower()
        if backend == "nebius":
            required.append("NEBIUS_API_KEY")
        elif backend == "openai":
            required.append("OPENAI_API_KEY")
        missing = [name for name in required if not os.getenv(name, "").strip() or os.getenv(name, "").startswith("REPLACE_WITH_")]
        if live and missing:
            raise ValueError("missing_configuration")
        stage = "policy loading"
        docs = load_corpus()
        profile = {}
        if bu != "Not specified":
            profile["bu"] = bu
        if known_fte:
            profile["fte"] = fte
        with st.spinner("Finding policy evidence and checking the answer…"):
            stage = "service setup"
            app = workflow(live, os.getenv("CHAT_MODEL", ""), os.getenv("LLM_BACKEND", ""), next(iter(docs.values())).metadata["source_hash"])
            # Reuse the original question when answering a clarification in preview mode.
            effective_question = question
            if not live and st.session_state.pending_question:
                effective_question = st.session_state.pending_question + "\nAdditional context: " + question
            stage = "answer workflow"
            result = app.invoke({"question": effective_question, "profile": profile, "history": previous})
        answer = result["answer"]
        paragraphs = [claim["text"] + "\n\nSources: " + ", ".join(claim["citations"]) for claim in answer["claims"]]
        if answer.get("clarification"):
            paragraphs.append(answer["clarification"])
        cited = list(dict.fromkeys(c for claim in answer["claims"] for c in claim["citations"]))
        if not live:
            cited = result.get("evidence", [])
        reply = {"role": "assistant", "content": "\n\n".join(paragraphs),
                 "sources": {key: docs[key].page_content for key in cited if key in docs},
                 "seconds": time.perf_counter() - start}
        st.session_state.pending_question = effective_question if result["status"] == "clarification" else None
    except Exception as exc:
        # Do not expose raw SDK exceptions, requests, credentials or internal traces.
        kind = type(exc).__name__
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        # Log allowlisted metadata only, never exception text or request headers.
        logging.getLogger("hr_chat").error("Request failed: stage=%s type=%s status=%s", stage, kind, status if isinstance(status, int) else "unknown")
        if stage == "configuration" and missing:
            detail = "Missing or placeholder settings: " + ", ".join(missing) + ". Open Manage app → Settings → Secrets, enter the real values, save and reboot. Your computer's .env is not uploaded to Streamlit Cloud."
        elif status == 401 or status == 403:
            detail = "A service rejected authentication or access. Check the Pinecone and Nebius keys and their project permissions in Streamlit Secrets."
        elif status == 404:
            detail = "A configured model or index was not found. Verify CHAT_MODEL is openai/gpt-oss-120b and the two Pinecone index names match your account."
        elif status == 429 or status == 402:
            detail = "A service reported a quota, billing or rate limit. Check the Nebius and Pinecone dashboards."
        elif isinstance(exc, RuntimeError) and str(exc).startswith("No matching indexed corpus"):
            detail = "Pinecone returned no records for this policy version. Verify the index names and PINECONE_NAMESPACE match the existing ingestion."
        elif kind in {"ValidationError", "OutputParserException", "OpenAIInvalidResponseError"}:
            detail = "The model response did not match the required answer format. Start a new conversation and retry; model output handling needs review if this repeats."
        elif "Connection" in kind or "Timeout" in kind:
            detail = "A service connection failed or timed out. Check the configured endpoints and service status, then retry."
        else:
            detail = "Open Manage app → Logs and share the diagnostic line beginning 'Request failed'. Do not share keys or request headers."
        reply = {"role": "assistant", "content": "I couldn't complete this request. " + detail +
                 f"\n\nDiagnostic: {stage} · {kind}" + (f" · HTTP {status}" if isinstance(status, int) else "")}
    st.session_state.messages.append(reply)
    render(reply)
