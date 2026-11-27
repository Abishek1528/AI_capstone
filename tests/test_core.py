"""Offline tests for the Hostel Helpdesk tools and retrieval.

These tests do not need an API key. Run them with:

    python -m pytest -q
or:
    python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval import build_retriever  # noqa: E402
from tools import days_until, draft_complaint, fee_due, guest_meal_cost  # noqa: E402


def test_fee_due_boundaries():
    assert fee_due(0)["total_fee"] == 0
    assert fee_due(-3)["total_fee"] == 0
    assert fee_due(3)["total_fee"] == 150      # 3 x 50
    assert fee_due(5)["total_fee"] == 250      # 5 x 50
    assert fee_due(8)["total_fee"] == 550      # 5x50 + 3x100
    assert fee_due(25)["total_fee"] == 2000    # capped
    assert fee_due(25)["cap_applied"] is True


def test_fee_due_rejects_bad_input():
    assert "error" in fee_due("many")


def test_guest_meal_cost():
    assert guest_meal_cost(2)["total_cost"] == 300
    assert guest_meal_cost(3, 2)["total_cost"] == 900
    assert "error" in guest_meal_cost(-1)


def test_draft_complaint_contains_key_facts():
    out = draft_complaint("Fan making noise", "B-214", "Electrical")
    assert "error" not in out
    assert "B-214" in out["draft"]
    assert "Fan making noise" in out["draft"]
    assert "Electrical" in out["draft"]


def test_draft_complaint_needs_inputs():
    assert "error" in draft_complaint("", "B-214")


def test_days_until_formats_and_errors():
    ok = days_until("2030-01-01")
    assert "days_difference" in ok and ok["days_difference"] > 0
    assert "error" in days_until("01-01-2030")


def test_retrieval_finds_the_right_document():
    retriever = build_retriever()

    # A short, ambiguous query must still land on the mess document.
    short = retriever.search("what time is dinner", k=3)
    assert short and "mess" in short[0].source.lower()

    mess = retriever.search("what time is dinner served", k=3)
    assert mess, "no results for mess timing"
    assert any("mess" in c.source.lower() for c in mess)

    fees = retriever.search("late fee if I pay 8 days late", k=3)
    assert fees
    assert "fee" in fees[0].source.lower()

    noise = retriever.search("quiet hours at night", k=3)
    assert noise
    assert "rule" in noise[0].source.lower()

    # Hyphenated terms should match their squashed form ("Wi-Fi" -> "wifi").
    wifi = retriever.search("what is the wifi password", k=3)
    assert wifi and "wi-fi" in wifi[0].heading.lower()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} tests passed.")
