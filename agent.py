"""
The Hostel Helpdesk agent.

One turn works in three visible steps:

1. ROUTE   - a small model call classifies the question into one of:
               "knowledge" (answer from the hostel documents),
               "tool"      (run one or more Python tools),
               "decline"   (out of scope or unsafe).
2. ACT     - the chosen action runs: retrieval + grounded answer, tool
               execution, or a polite refusal.
3. ANSWER  - the final reply, always grounded in the retrieved documents or
               the tool output, with the sources that were used.

Conversation memory is kept by the caller as a list of gemini `Content`
objects, so follow-up questions such as "and if it is 10 days late?" work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from google import genai
from google.genai import types

from retrieval import Retriever
from tools import TOOLS, TOOL_DESCRIPTIONS

DEFAULT_MODEL = "gemini-2.5-flash"

MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

ROUTES = ("knowledge", "tool", "decline")

_HOSTEL_NAME = "Green Valley Students' Hostel"

_ROUTER_SYSTEM = f"""You are the routing brain of a helpdesk assistant for the \
{_HOSTEL_NAME}. Your ONLY job is to decide how to handle the user's latest message.

The assistant may ONLY:
  - answer questions about this hostel (rules, mess timings and menu, facilities,
    complaints and maintenance, fees/deposit/fines, emergency and contacts), using
    the hostel's own documents;
  - run a small set of Python tools when a calculation or a draft is needed;
  - politely decline anything else.

Available tools:
  - fee_due(days_late): compute the late-payment fee from the number of late days.
  - guest_meal_cost(guests, meals): compute guest-meal cost.
  - draft_complaint(issue, room_number, category): draft a complaint message.
  - days_until(target_date): days between today and a date (YYYY-MM-DD).

Choose the route:
  "tool"      - the request needs a calculation or a generated draft. Provide the
                tool call(s) with concrete arguments. Fill arguments from the
                conversation when the user is following up.
  "knowledge" - the request is a question about the hostel that the documents
                should answer. No tool is needed.
  "decline"   - the request is outside the hostel's job OR is unsafe/inappropriate.
                Examples: medical or legal advice, another person's private data,
                exam answers, general knowledge, coding help, anything not about
                this hostel. When unsure but the topic is still about hostel life,
                prefer "knowledge" so the documents can decide.

Reply with ONLY a JSON object (no markdown fences) in exactly this shape:
{{
  "route": "knowledge" | "tool" | "decline",
  "tool_calls": [ {{"name": "<tool name>", "args": {{...}} }} ],
  "reason": "<one short sentence>"
}}
For the "knowledge" and "decline" routes, "tool_calls" must be an empty list.
"""

_ANSWER_SYSTEM = f"""You are the {_HOSTEL_NAME} helpdesk assistant. You are warm, \
brief and precise.

Hard rules:
- Answer ONLY from the CONTEXT (hostel documents) and the TOOL RESULTS given to you.
- If the context does not contain the answer, say plainly that the hostel documents
  do not cover it. Never invent rules, timings, amounts or phone numbers.
- Cite the sources you used with bracketed numbers like [1], [2] that match the
  numbered sources in the CONTEXT. If you use no source, do not cite anything.
- Use exact figures from the documents. Follow the hostel's own wording for
  procedures.
- Keep answers short: a few sentences or a short list. Prefer concrete numbers and
  steps over general advice.
- You are not a doctor, lawyer or financial adviser. If asked for such advice, say
  you cannot help and point the user to the appropriate desk or professional.
"""


@dataclass
class TurnResult:
    """Everything the UI needs to render one turn."""

    route: str
    reason: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def make_client(api_key: str) -> genai.Client:
    """Create a Gemini client from a user-supplied API key."""
    if not api_key or not api_key.strip():
        raise ValueError("A Gemini API key is required.")
    return genai.Client(api_key=api_key.strip())


def _to_contents(history: Iterable[dict[str, str]]) -> list[types.Content]:
    """Convert lightweight {role, text} turns into gemini Content objects."""
    contents: list[types.Content] = []
    for turn in history:
        role = "model" if turn.get("role") in ("model", "assistant") else "user"
        text = turn.get("text", "")
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def _history_digest(history: list[dict[str, str]], max_turns: int = 6) -> str:
    if not history:
        return "(no earlier conversation)"
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        who = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {turn.get('text', '').strip()}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating stray fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _fallback_route(question: str) -> dict[str, Any]:
    """Keyword fallback when the router reply cannot be parsed."""
    q = (question or "").lower()
    if any(w in q for w in ("fee", "fine", "late", "penalty", "due")):
        return {"route": "knowledge", "tool_calls": [], "reason": "fallback: fee topic"}
    if any(w in q for w in ("complaint", "draft", "write a")):
        return {"route": "knowledge", "tool_calls": [], "reason": "fallback: draft topic"}
    return {"route": "knowledge", "tool_calls": [], "reason": "fallback: default to documents"}


def route_query(
    client: genai.Client,
    question: str,
    history: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Classify the question into a route (and tool calls) with one model call."""
    prompt = (
        f"Earlier conversation:\n{_history_digest(history)}\n\n"
        f"User's latest message: {question}\n\n"
        "Decide the route and reply with the JSON object only."
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_ROUTER_SYSTEM,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        decision = _extract_json(response.text or "")
    except Exception:
        decision = _fallback_route(question)

    route = str(decision.get("route", "knowledge")).lower().strip()
    if route not in ROUTES:
        route = "knowledge"
    tool_calls = decision.get("tool_calls") or []
    if route != "tool":
        tool_calls = []
    return {
        "route": route,
        "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        "reason": str(decision.get("reason", "")).strip() or "No reason given.",
    }


def _format_context(chunks: list) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved chunks as a numbered context block plus a sources list."""
    if not chunks:
        return "(no matching sections were found in the hostel documents)", []
    blocks, sources = [], []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] Source: {chunk.source} | Section: {chunk.heading}\n{chunk.text}")
        sources.append(
            {
                "index": i,
                "source": chunk.source,
                "heading": chunk.heading,
                "snippet": chunk.snippet,
                "score": round(chunk.score, 3),
            }
        )
    return "\n\n".join(blocks), sources


def execute_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the requested tools and record a step (name, args, result) for each."""
    steps: list[dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get("name", "")).strip()
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if name not in TOOLS:
            steps.append(
                {"name": name or "(unknown)", "args": args, "result": {"error": "Unknown tool."}}
            )
            continue
        try:
            result = TOOLS[name](**args)
        except Exception as exc:  # keep the turn alive on a bad argument
            result = {"error": f"{type(exc).__name__}: {exc}"}
        steps.append({"name": name, "args": args, "result": result})
    return steps


def _generate_answer(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Generate the final answer, streaming tokens to on_token if provided."""
    config = types.GenerateContentConfig(
        system_instruction=_ANSWER_SYSTEM,
        temperature=0.2,
    )
    if on_token is None:
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return (response.text or "").strip()

    collected: list[str] = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        piece = chunk.text or ""
        if piece:
            collected.append(piece)
            on_token(piece)
    return "".join(collected).strip()


def run_turn(
    client: genai.Client,
    retriever: Retriever,
    question: str,
    history: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    top_k: int = 4,
    on_token: Callable[[str], None] | None = None,
    on_route: Callable[[dict[str, Any]], None] | None = None,
) -> TurnResult:
    """Run one full turn: route, act, answer. Returns a TurnResult."""
    question = (question or "").strip()
    if not question:
        return TurnResult(route="decline", reason="Empty question.", answer="Please type a question about the hostel.")

    decision = route_query(client, history=history, question=question, model=model)
    route = decision["route"]
    if on_route is not None:
        on_route(decision)

    # --- Route: decline -------------------------------------------------
    if route == "decline":
        answer = (
            "Sorry, that is outside what I can help with. I am the helpdesk assistant for "
            f"{_HOSTEL_NAME}, so I can only answer questions about hostel rules, the mess and "
            "facilities, complaints and maintenance, fees and fines, and emergencies. "
            "For anything else, please contact the Hostel Office or the relevant desk."
        )
        return TurnResult(route="decline", reason=decision["reason"], answer=answer)

    # --- Route: knowledge and tool both retrieve context -----------------
    chunks = retriever.search(question, k=top_k)
    context_block, sources = _format_context(chunks)

    # --- Route: tool -----------------------------------------------------
    tool_steps: list[dict[str, Any]] = []
    tool_block = "(no tools were used)"
    if route == "tool":
        tool_steps = execute_tool_calls(decision["tool_calls"])
        if not tool_steps:
            # Router said "tool" but named none: fall back to a knowledge answer.
            route = "knowledge"
        else:
            rendered = []
            for step in tool_steps:
                rendered.append(
                    f"Tool: {step['name']}\nArguments: {json.dumps(step['args'])}\n"
                    f"Result: {json.dumps(step['result'], ensure_ascii=False, default=str)}"
                )
            tool_block = "\n\n".join(rendered)

    # --- Compose the answer prompt and generate --------------------------
    prompt = (
        f"CONTEXT (hostel documents):\n{context_block}\n\n"
        f"TOOL RESULTS:\n{tool_block}\n\n"
        f"User's question: {question}\n\n"
        "Write the final answer for the user now, following the hard rules."
    )
    contents = _to_contents(history) + [types.Content(role="user", parts=[types.Part(text=prompt)])]

    try:
        answer = _generate_answer(client, model, contents, on_token=on_token)
    except Exception as exc:
        return TurnResult(
            route=route,
            reason=decision["reason"],
            answer="I could not reach the model just now. Please check your API key and try again.",
            sources=sources,
            tool_calls=tool_steps,
            error=f"{type(exc).__name__}: {exc}",
        )

    return TurnResult(
        route=route,
        reason=decision["reason"],
        answer=answer or "I do not have an answer for that.",
        sources=sources,
        tool_calls=tool_steps,
    )
