# 🏠 Hostel Helpdesk — an AI assistant for hostel residents

An end-to-end AI solution for **one clear user** (a hostel resident) and **one clear job**
(get correct, sourced answers about hostel life, plus the small sums and drafts that go with
it).

The assistant answers from the hostel's own documents, **shows the sources it used**, runs
**Python tools** for calculations and drafts, **politely declines** anything outside its job,
**remembers the conversation**, and ships with **evidence** (`eval.md` + a runnable harness).

> **Live app:** `https://your-app-name.streamlit.app` ← replace after deploying (see below).
> Every visitor pastes **their own free Gemini key** in the sidebar. No key is stored, logged
> or committed anywhere in this repository.

---

## 1. The user and the job

| | |
| --- | --- |
| **User** | A resident of *Green Valley Students' Hostel* (fictional). |
| **Job to be done** | "Tell me the hostel's rule, timing, fee or process — correctly, with the source — and do the small sums and drafts I would otherwise do by hand." |
| **Out of scope** | Medical, legal and exam advice, other people's private data, and anything not about this hostel. The assistant says so and points to the right desk. |

---

## 2. What it does (mapped to the brief)

| Requirement | How it is met |
| --- | --- |
| Answers questions from **your own documents** | BM25 retrieval over `knowledge_base/*.md`, then a grounded Gemini answer (`retrieval.py`, `agent.py`). |
| **Shows its sources** | Every answer lists the retrieved sections (`file · heading · snippet · score`) in a *Sources* panel. |
| **Uses tools you wrote** | `fee_due`, `guest_meal_cost`, `draft_complaint`, `days_until` in `tools.py`, chosen by the router and executed in Python. |
| **Politely declines** | A `decline` route with a friendly, fixed refusal. |
| **Remembers the conversation** | Chat history is kept as Gemini `Content` turns and re-sent each turn. |
| **Comes with evidence** | `eval.md` (12 cases) + `run_eval.py` (reproducible harness) + `tests/test_core.py` (offline tests). |
| **Visitor's own key** | A sidebar `password` field; the key lives only in the browser session. |
| **Visible route / sources / tool steps** | The UI shows a **route badge** (why the assistant chose to answer / use a tool / decline), a **Sources** expander and a **Tool steps** expander on every reply. |

---

## 3. Knowledge base (5 documents, written for this project)

All documents are **original writing created for this capstone** — no private, copyrighted or
personal material is used. The hostel, its people, phone numbers and amounts are fictional.

| File | Subject |
| --- | --- |
| `knowledge_base/01_hostel_rules.md` | Rules, gate timings, quiet hours, guests, prohibited items, discipline |
| `knowledge_base/02_mess_and_facilities.md` | Mess timings and menu, guest meals, laundry, Wi-Fi, gym, study hall |
| `knowledge_base/03_complaints_and_maintenance.md` | How to complain, categories, service levels, escalation |
| `knowledge_base/04_fees_and_fines.md` | Fees, deposit, **late-fee policy**, fine schedule, refunds |
| `knowledge_base/05_emergency_and_contacts.md` | Emergency procedures, safety, contacts |

Each file carries a provenance note stating it is original fictional content.

---

## 4. How a turn works

```
question
   │
   ▼
┌─────────────┐   one small model call, returns JSON
│ 1. ROUTE    │──► { "route": knowledge | tool | decline, "tool_calls": [...], "reason": ... }
└─────────────┘
   │
   ├── knowledge ─► retrieve top-k sections ─┐
   ├── tool ──────► run Python tool(s) ──────┤
   │                                        ▼
   │                              ┌──────────────────┐
   │                              │ 2. ANSWER        │  grounded Gemini call
   │                              │  (streamed)      │  cites [1], [2]
   │                              └──────────────────┘
   └── decline ──► fixed polite refusal (no model call)
```

The **route**, the **sources** and the **tool steps** are all shown in the UI, so the mentor can
see *why* an answer was given.

### Routes

| Route | Meaning |
| --- | --- |
| `knowledge` | Question about the hostel; answered from the retrieved documents. |
| `tool` | Needs a calculation or a draft; one or more Python tools run first. |
| `decline` | Outside the job or unsafe; a polite refusal is returned. |

### Tools (`tools.py`)

| Tool | Purpose |
| --- | --- |
| `fee_due(days_late)` | Late fee: ₹50/day for days 1–5, ₹100/day from day 6, capped at ₹2,000. |
| `guest_meal_cost(guests, meals)` | Guest meal cost at ₹150 per person per meal. |
| `draft_complaint(issue, room_number, category)` | Ready-to-send complaint message. |
| `days_until(target_date)` | Days between today and a date (`YYYY-MM-DD`). |

The tools own all the arithmetic and the draft text, so the model never invents a number or a
template.

---

## 5. Project structure

```
.
├── app.py                     # Streamlit UI (sidebar key, chat, route/sources/tools panels)
├── agent.py                   # routing, retrieval-grounded answering, tool execution, memory
├── retrieval.py               # chunking + pure-Python BM25 retriever
├── tools.py                   # the Python tools
├── run_eval.py                # 12-case evaluation harness (live) + offline check
├── eval.md                    # the 12 test cases, results and the one fix
├── tests/test_core.py         # offline unit tests (no API key needed)
├── knowledge_base/            # the 5 hostel documents
├── requirements.txt
└── .streamlit/config.toml
```

---

## 6. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then paste your own Gemini key (free at <https://aistudio.google.com/app/apikey>) into the
sidebar. Python 3.10+ is required.

Offline checks (no key needed):

```bash
python tests/test_core.py        # unit tests
python run_eval.py --offline     # knowledge-base + retrieval sanity check
```

Live evaluation with your key:

```bash
# Windows
set GEMINI_API_KEY=your_key_here && python run_eval.py
# macOS / Linux
GEMINI_API_KEY=your_key_here python run_eval.py
```

---

## 7. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to <https://share.streamlit.io> → **New app**.
3. Pick this repository, branch `main`, and **main file path** `app.py`.
4. Deploy. No secrets need to be configured — the visitor supplies their own key.
5. Open the app in a **private/incognito window**, paste a key, ask a question, and confirm
   the route, sources and tool steps are visible.

---

## 8. Limits and honesty

- The assistant answers **only** from the five documents. If they do not cover something, it
  says so (see case **K6** in `eval.md`) instead of guessing.
- Retrieval is **lexical (BM25)**, not embedding-based. It is fast, offline and reproducible,
  and it is very good for this small, keyword-rich corpus; it would need embeddings to scale to
  a much larger, paraphrased corpus.
- The tool figures come from `04_fees_and_fines.md`. If the hostel changes a fee, both the
  document and the constants in `tools.py` must be updated together.
- The phone numbers in `05_emergency_and_contacts.md` are **placeholders** and must not be
  dialled.

---

## 9. No secrets

There is **no API key anywhere in this repository**. The key is entered by the visitor at
runtime and held only in the Streamlit session. `.gitignore` excludes `.env`,
`.streamlit/secrets.toml` and key files.
