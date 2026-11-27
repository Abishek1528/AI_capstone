"""
Tool implementations for the Hostel Helpdesk.

These are plain Python functions. The assistant decides *when* to call a tool,
but the tool itself does all the arithmetic and text assembly, so the numbers
and the drafts are never invented by the model.

Policy references (see knowledge_base/04_fees_and_fines.md):
  - Late fee, days 1-5 late:  Rs.50 per day
  - Late fee, day 6 onward:   Rs.100 per day
  - Maximum late fee:         Rs.2000 per late payment
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

LATE_FEE_TIER1_RATE = 50      # Rs per day, days 1-5
LATE_FEE_TIER2_RATE = 100     # Rs per day, day 6 onward
LATE_FEE_TIER1_DAYS = 5
LATE_FEE_CAP = 2000           # Rs, per late payment

GUEST_MEAL_RATE = 150         # Rs per person per meal


def fee_due(days_late: int) -> dict[str, Any]:
    """Calculate the late-payment fee for a monthly hostel bill.

    Args:
        days_late: How many days the payment is late. 0 or less means no fee.

    Returns:
        A dictionary with the number of late days, a breakdown of the tiers,
        and the total late fee in rupees.
    """
    try:
        days = int(days_late)
    except (TypeError, ValueError):
        return {"error": f"days_late must be a whole number, got {days_late!r}"}

    if days <= 0:
        return {
            "days_late": max(days, 0),
            "tier_1_days": 0,
            "tier_1_amount": 0,
            "tier_2_days": 0,
            "tier_2_amount": 0,
            "total_fee": 0,
            "cap_applied": False,
            "note": "No fee is due when the payment is on time or early.",
        }

    tier_1_days = min(days, LATE_FEE_TIER1_DAYS)
    tier_2_days = max(days - LATE_FEE_TIER1_DAYS, 0)
    tier_1_amount = tier_1_days * LATE_FEE_TIER1_RATE
    tier_2_amount = tier_2_days * LATE_FEE_TIER2_RATE

    subtotal = tier_1_amount + tier_2_amount
    total = min(subtotal, LATE_FEE_CAP)

    return {
        "days_late": days,
        "tier_1_days": tier_1_days,
        "tier_1_amount": tier_1_amount,
        "tier_2_days": tier_2_days,
        "tier_2_amount": tier_2_amount,
        "subtotal": subtotal,
        "total_fee": total,
        "cap_applied": subtotal > LATE_FEE_CAP,
        "note": (
            f"Late fee is capped at Rs.{LATE_FEE_CAP} per late payment."
            if subtotal > LATE_FEE_CAP
            else "Late fee follows the Rs.50/day (days 1-5) then Rs.100/day policy."
        ),
    }


def guest_meal_cost(guests: int, meals: int = 1) -> dict[str, Any]:
    """Calculate the cost of guest meals at the hostel mess.

    Args:
        guests: Number of guest persons eating.
        meals: Number of meals per guest (default 1).

    Returns:
        A dictionary with the flat rate, the count, and the total cost.
    """
    try:
        g = int(guests)
        m = int(meals)
    except (TypeError, ValueError):
        return {"error": "guests and meals must be whole numbers."}

    if g < 0 or m < 0:
        return {"error": "guests and meals cannot be negative."}

    return {
        "rate_per_meal": GUEST_MEAL_RATE,
        "guests": g,
        "meals_per_guest": m,
        "total_meals": g * m,
        "total_cost": g * m * GUEST_MEAL_RATE,
    }


def draft_complaint(issue: str, room_number: str, category: str = "General") -> dict[str, Any]:
    """Draft a formal complaint message for the hostel maintenance desk.

    Args:
        issue: A short description of the problem, in the resident's words.
        room_number: The resident's room, e.g. "B-214".
        category: Complaint category such as Electrical, Plumbing, Carpentry,
            Housekeeping, Mess, Internet, Safety or Other.

    Returns:
        A dictionary containing the ready-to-send draft text.
    """
    issue = (issue or "").strip()
    room_number = (room_number or "").strip()
    category = (category or "General").strip() or "General"

    if not issue:
        return {"error": "An issue description is required to draft a complaint."}
    if not room_number:
        return {"error": "A room number is required to draft a complaint."}

    today = date.today().strftime("%d %B %Y")
    draft = (
        "To: Maintenance Desk, Green Valley Students' Hostel\n"
        f"Date: {today}\n"
        f"Category: {category}\n"
        f"Room: {room_number}\n\n"
        "Dear Maintenance Team,\n\n"
        f"I would like to report the following issue in room {room_number}:\n"
        f'    "{issue}"\n\n'
        "Please arrange for this to be looked into at the earliest. Kindly share a "
        "ticket number so that I can track the status of the complaint.\n\n"
        "Thank you,\n"
        "[Your name]\n"
        f"Room {room_number}, Green Valley Students' Hostel\n"
        "[Hostel ID and contact number]"
    )
    return {
        "category": category,
        "room_number": room_number,
        "draft": draft,
        "reminder": (
            "Replace the placeholders in square brackets with your name, Hostel ID "
            "and contact number before sending."
        ),
    }


def days_until(target_date: str) -> dict[str, Any]:
    """Calculate how many days remain until a given date (or since it passed).

    Args:
        target_date: A date in YYYY-MM-DD format, e.g. "2026-12-05".

    Returns:
        A dictionary with today's date, the target date, the signed day
        difference, and a human-readable phrase.
    """
    try:
        target = datetime.strptime(str(target_date).strip(), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return {
            "error": (
                f"Could not read the date {target_date!r}. "
                "Please use the YYYY-MM-DD format, for example 2026-12-05."
            )
        }

    today = date.today()
    delta = (target - today).days

    if delta > 0:
        phrase = f"{delta} day(s) from today, on {target.strftime('%d %B %Y')}."
    elif delta == 0:
        phrase = f"That date is today ({target.strftime('%d %B %Y')})."
    else:
        phrase = f"That date was {abs(delta)} day(s) ago, on {target.strftime('%d %B %Y')}."

    return {
        "today": today.isoformat(),
        "target_date": target.isoformat(),
        "days_difference": delta,
        "phrase": phrase,
    }


# Registry the agent uses to look up a tool by name.
TOOLS: dict[str, Any] = {
    "fee_due": fee_due,
    "guest_meal_cost": guest_meal_cost,
    "draft_complaint": draft_complaint,
    "days_until": days_until,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "fee_due": "Compute the late-payment fee. Args: days_late (int).",
    "guest_meal_cost": "Compute guest meal cost. Args: guests (int), meals (int, default 1).",
    "draft_complaint": (
        "Draft a complaint message. Args: issue (str), room_number (str), "
        "category (str, optional)."
    ),
    "days_until": "Days until/from a date. Args: target_date (str, YYYY-MM-DD).",
}
