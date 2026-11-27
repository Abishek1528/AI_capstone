"""
Green Valley Students' Hostel — Helpdesk Assistant (Streamlit app).

A retrieval-augmented assistant that answers questions from the hostel's own
documents, shows its sources, uses small Python tools for calculations and
drafts, remembers the conversation, and politely declines anything outside its
job.

The visitor pastes their OWN free Gemini API key in the sidebar. The key is used
only from the running session and is never stored, logged, or committed.
"""

from __future__ import annotations

import streamlit as st

from agent import DEFAULT_MODEL, MODEL_CHOICES, make_client, run_turn
from retrieval import build_retriever

st.set_page_config(
    page_title="Hostel Helpdesk",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

ROUTE_BADGE = {
    "knowledge": ("📚", "knowledge", "Answered from the hostel documents"),
    "tool": ("🛠", "tool", "Used a Python tool"),
    "decline": ("🚫", "decline", "Out of scope — politely declined"),
}

EXAMPLE_QUESTIONS = [
    "What time is dinner served in the mess?",
    "If my hostel fee is 8 days late, how much late fee do I pay?",
    "Draft a complaint about a leaking tap in room B-214.",
    "Can my cousin stay overnight in my room?",
    "How many days until 2026-12-05?",
    "What is the Wi-Fi password of the college next door?",
]


@st.cache_resource(show_spinner=False)
def get_retriever():
    """Load and index the knowledge base once per server process."""
    return build_retriever()


def init_state() -> None:
    st.session_state.setdefault("turns", [])        # what we render
    st.session_state.setdefault("history", [])      # memory sent to the model
    st.session_state.setdefault("pending", None)    # example button hand-off


def route_badge(route: str, reason: str) -> str:
    icon, label, blurb = ROUTE_BADGE.get(route, ("❔", route, ""))
    return f"{icon} **Route: `{label}`** — {blurb}.  \n<small>Why: {reason}</small>"


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(
                f"**[{src['index']}] {src['source']}** · {src['heading']} "
                f"<small>(score {src['score']})</small>",
                unsafe_allow_html=True,
            )
            st.caption(src["snippet"])
            st.markdown("")


def render_tools(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🛠 Tool steps ({len(tool_calls)})"):
        for i, step in enumerate(tool_calls, start=1):
            st.markdown(f"**Step {i}: `{step['name']}`**")
            st.code(f"arguments = {step['args']}", language="python")
            result = step["result"]
            if isinstance(result, dict) and "draft" in result:
                st.text(result["draft"])
            else:
                st.json(result)


def render_assistant_turn(turn: dict) -> None:
    st.markdown(route_badge(turn.get("route", "knowledge"), turn.get("reason", "")))
    st.markdown(turn.get("answer", ""))
    if turn.get("error"):
        st.error(turn["error"])
    render_sources(turn.get("sources", []))
    render_tools(turn.get("tool_calls", []))


# --------------------------------------------------------------------------
# Sidebar: key, model, help
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🏠 Hostel Helpdesk")
    st.caption("Green Valley Students' Hostel")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Paste your own free key",
        help="Get a free key at aistudio.google.com/app/apikey. "
        "It stays in this browser session and is never stored.",
    )
    model = st.selectbox(
        "Model",
        MODEL_CHOICES,
        index=MODEL_CHOICES.index(DEFAULT_MODEL),
        help="gemini-2.5-flash is a good default.",
    )

    if api_key:
        st.success("Key entered for this session.")
    else:
        st.info("Paste a key to start chatting.")

    st.divider()
    st.markdown(
        "**This assistant can**\n"
        "- Answer from the hostel rulebooks, mess guide, complaint process, "
        "fee policy and emergency info\n"
        "- Do sums (late fees, guest meal cost, days to a date)\n"
        "- Draft a complaint message\n\n"
        "**It will decline** anything outside hostel life — medical, legal or "
        "exam advice, other people's private data."
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["turns"] = []
        st.session_state["history"] = []
        st.rerun()
    st.caption("No key is stored on the server. You bring your own.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
init_state()
retriever = get_retriever()
doc_count = len({c.source for c in retriever.chunks})

st.title("Hostel Helpdesk")
st.caption(
    f"Ask about hostel rules, the mess, complaints, fees, or emergencies. "
    f"Answers come from {doc_count} hostel documents and show their sources."
)

if not st.session_state["turns"]:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending"] = example
            st.rerun()

for turn in st.session_state["turns"]:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["text"])
    else:
        with st.chat_message("assistant"):
            render_assistant_turn(turn)

prompt = st.chat_input("Ask about the hostel…")

if st.session_state["pending"]:
    prompt = st.session_state["pending"]
    st.session_state["pending"] = None

if prompt:
    if not api_key:
        st.warning("Please paste your Gemini API key in the sidebar first.")
    else:
        st.session_state["turns"].append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            badge_holder = st.container()
            answer_holder = st.empty()
            collected: list[str] = []

            def on_route(decision: dict) -> None:
                with badge_holder:
                    st.markdown(route_badge(decision["route"], decision["reason"]))

            def on_token(token: str) -> None:
                collected.append(token)
                answer_holder.markdown("".join(collected))

            try:
                client = make_client(api_key)
                result = run_turn(
                    client=client,
                    retriever=retriever,
                    question=prompt,
                    history=st.session_state["history"],
                    model=model,
                    on_token=on_token,
                    on_route=on_route,
                )
            except Exception as exc:  # bad key, network error, etc.
                result = None
                st.error(f"Something went wrong: {exc}")
                st.session_state["turns"].append(
                    {
                        "role": "assistant",
                        "route": "decline",
                        "reason": "The request could not be completed.",
                        "answer": "I could not complete that. Please check your API key and try again.",
                        "sources": [],
                        "tool_calls": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if result is not None:
            st.session_state["turns"].append(
                {
                    "role": "assistant",
                    "route": result.route,
                    "reason": result.reason,
                    "answer": result.answer,
                    "sources": result.sources,
                    "tool_calls": result.tool_calls,
                    "error": result.error,
                }
            )
            st.session_state["history"].append({"role": "user", "text": prompt})
            st.session_state["history"].append({"role": "assistant", "text": result.answer})

        st.rerun()
