"""The agent's skills (tools). Each tool = name + argument schema + handler (+ optional precheck).

Skills
  policy_qa            read   Answer a policy question via the guarded PolicyBot (RAG).
  check_expense        read   Deterministic compliance check against expense policy limits.
  submit_leave_request WRITE  Record a leave request in the ledger (needs human confirmation).

Design note: handlers are deliberately simple and trust their input. Validation lives in the
action guard (guardrails/action_guard.py), so tests can prove the guard is what protects us.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

# Facts the tools rely on. tests/test_agent_tools.py asserts they still match the policy document.
MEAL_PER_DAY = 1500
HOTEL_PER_NIGHT = 8000
DIRECTOR_APPROVAL_ABOVE = 25000
LEAVE_ALLOWANCE = {"casual": 12, "sick": 8, "earned": 18}
MIN_NOTICE_WORKING_DAYS = 3


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    handler: Callable[[dict], ToolResult]
    writes: bool = False
    precheck: Callable[[dict], str | None] | None = None


class ToolRegistry(dict):
    """name -> Tool (a dict, so `name in registry` is the allow-list check)."""

    def add(self, tool: Tool) -> "ToolRegistry":
        self[tool.name] = tool
        return self

    def catalog(self) -> str:
        lines = []
        for t in self.values():
            args = ", ".join(f"{k}:{v['type']}" for k, v in t.params.items())
            lines.append(f"- {t.name}({args}){' [WRITE]' if t.writes else ''}: {t.description}")
        return "\n".join(lines)


def money(x) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def fmt_date(d: date) -> str:
    # Written as '02 Nov 2026' (not 2026-11-02): the existing output guard's phone regex
    # false-positives on ISO dates. See docs/AGENT.md "Finding".
    return d.strftime("%d %b %Y")


def parse_iso(s: str) -> date | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def working_days_between(today: date, start: date) -> int:
    """Weekdays strictly between today and the start date (our reading of 'N working days in advance')."""
    n, d = 0, today + timedelta(days=1)
    while d < start:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


class LeaveLedger:
    """In-memory system of record. The tests inspect it to prove what the agent really did."""

    def __init__(self):
        self.requests: list[dict] = []

    def used(self, leave_type: str) -> int:
        return sum(r["days"] for r in self.requests if r["leave_type"] == leave_type)

    def remaining(self, leave_type: str) -> int:
        return LEAVE_ALLOWANCE[leave_type] - self.used(leave_type)

    def add(self, leave_type: str, days: int, start: date) -> str:
        rid = f"LR-{len(self.requests) + 1:04d}"
        self.requests.append({"id": rid, "leave_type": leave_type, "days": days, "start": start})
        return rid

    def overlaps(self, start: date, days: int) -> bool:
        end = start + timedelta(days=days - 1)
        return any(start <= r["start"] + timedelta(days=r["days"] - 1) and r["start"] <= end
                   for r in self.requests)


def build_registry(bot, ledger: LeaveLedger, today: date) -> ToolRegistry:
    # ---- policy_qa --------------------------------------------------------------------
    def policy_qa(args):
        r = bot.ask(args["question"])
        data = {"bot_status": r.status}
        if r.status in {"blocked_input", "blocked_output"}:
            data["terminal_status"] = r.status          # the agent halts the whole run on this
        return ToolResult(r.status in {"answered", "refused_unknown"}, r.answer, data)

    # ---- check_expense ----------------------------------------------------------------
    def check_expense(args):
        cat, amt = args["category"], args["amount"]
        if cat == "meal_per_day":
            ok, msg = amt <= MEAL_PER_DAY, (
                f"INR {money(amt)} is within the meal allowance of INR {money(MEAL_PER_DAY)} per day."
                if amt <= MEAL_PER_DAY else
                f"INR {money(amt)} exceeds the meal allowance of INR {money(MEAL_PER_DAY)} per day.")
        elif cat == "hotel_per_night":
            ok, msg = amt <= HOTEL_PER_NIGHT, (
                f"INR {money(amt)} is within the hotel limit of INR {money(HOTEL_PER_NIGHT)} per night."
                if amt <= HOTEL_PER_NIGHT else
                f"INR {money(amt)} exceeds the hotel limit of INR {money(HOTEL_PER_NIGHT)} per night.")
        else:
            ok = amt <= DIRECTOR_APPROVAL_ABOVE
            msg = (f"INR {money(amt)} does not need director approval (required above "
                   f"INR {money(DIRECTOR_APPROVAL_ABOVE)})." if ok else
                   f"INR {money(amt)} needs director approval (required above "
                   f"INR {money(DIRECTOR_APPROVAL_ABOVE)}).")
        return ToolResult(True, msg, {"within_policy": ok})

    # ---- submit_leave_request (WRITE) -------------------------------------------------
    def leave_precheck(args):
        start = parse_iso(args["start_date"])
        if start is None:
            return "start_date is not a valid date (use YYYY-MM-DD)"
        if start < today:
            return "start_date is in the past"
        lt, days = args["leave_type"], args["days"]
        if lt != "sick":
            notice = working_days_between(today, start)
            if notice < MIN_NOTICE_WORKING_DAYS:
                return (f"leave must be requested at least {MIN_NOTICE_WORKING_DAYS} working days in "
                        f"advance (only {notice} before {fmt_date(start)})")
        if days > ledger.remaining(lt):
            return f"only {ledger.remaining(lt)} day(s) of {lt} leave remaining (annual limit {LEAVE_ALLOWANCE[lt]})"
        if ledger.overlaps(start, days):
            return "this overlaps an existing leave request"
        return None

    def submit_leave(args):
        start = parse_iso(args["start_date"])
        rid = ledger.add(args["leave_type"], args["days"], start)
        msg = (f"Leave request {rid} submitted: {args['days']} day(s) of {args['leave_type']} leave "
               f"from {fmt_date(start)}.")
        if args["leave_type"] == "sick" and args["days"] > 2:
            msg += " A medical certificate is required for sick leave longer than 2 consecutive days."
        return ToolResult(True, msg, {"request_id": rid})

    return (ToolRegistry()
            .add(Tool("policy_qa", "Answer a question about the Acme policy handbook.",
                      {"question": {"type": "str", "max_len": 1000}}, policy_qa))
            .add(Tool("check_expense", "Check an amount against expense policy limits.",
                      {"category": {"type": "str", "enum": ["meal_per_day", "hotel_per_night", "single_claim"]},
                       "amount": {"type": "number", "min": 0.01, "max": 10_000_000}}, check_expense))
            .add(Tool("submit_leave_request", "Submit a leave request (state-changing).",
                      {"leave_type": {"type": "str", "enum": sorted(LEAVE_ALLOWANCE)},
                       "days": {"type": "int", "min": 1, "max": 30},
                       "start_date": {"type": "str", "max_len": 10}},
                      submit_leave, writes=True, precheck=leave_precheck)))
