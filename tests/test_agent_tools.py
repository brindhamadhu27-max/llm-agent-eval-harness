"""Skills: the tools' own logic, boundaries, and that their hard-coded facts match the policy document."""
from datetime import date

import pytest

from agent import tools as T
from agent.factory import FIXED_TODAY, build_agent
from guardrails.action_guard import validate_args


def test_tool_constants_match_the_policy_document():
    """If someone edits the policy, the tools must not silently drift from it."""
    doc = T.__file__ and open("app/data/company_policy.md", encoding="utf-8").read()
    for fact in [f"{T.MEAL_PER_DAY:,}", f"{T.HOTEL_PER_NIGHT:,}", f"{T.DIRECTOR_APPROVAL_ABOVE:,}",
                 f"{T.LEAVE_ALLOWANCE['casual']} days of casual", f"{T.LEAVE_ALLOWANCE['sick']} days of sick",
                 f"{T.LEAVE_ALLOWANCE['earned']} days of earned", f"{T.MIN_NOTICE_WORKING_DAYS} working days"]:
        assert fact in doc, fact


@pytest.mark.parametrize("amount,within", [(1500, True), (1500.01, False), (0.01, True)])
def test_meal_boundary(amount, within):
    _, _, reg = build_agent()
    assert reg["check_expense"].handler({"category": "meal_per_day", "amount": amount}).data["within_policy"] is within


@pytest.mark.parametrize("amount,within", [(8000, True), (8001, False)])
def test_hotel_boundary(amount, within):
    _, _, reg = build_agent()
    assert reg["check_expense"].handler({"category": "hotel_per_night", "amount": amount}).data["within_policy"] is within


@pytest.mark.parametrize("amount,needs", [(25000, False), (25001, True)])
def test_director_approval_boundary(amount, needs):
    _, _, reg = build_agent()
    r = reg["check_expense"].handler({"category": "single_claim", "amount": amount})
    assert ("needs director approval" in r.message) is needs


@pytest.mark.parametrize("args,why", [
    ("not a dict", "object"),
    ({"question": "x", "extra": 1}, "unexpected"),
    ({}, "missing"),
    ({"question": 5}, "string"),
    ({"question": "   "}, "empty"),
    ({"question": "x" * 1001}, "longer"),
])
def test_validate_args_rejects_bad_policy_qa_args(args, why):
    _, _, reg = build_agent()
    err = validate_args(reg["policy_qa"].params, args)
    assert err and why in err


@pytest.mark.parametrize("args", [
    {"leave_type": "unpaid", "days": 1, "start_date": "2026-11-02"},     # not in enum
    {"leave_type": "casual", "days": True, "start_date": "2026-11-02"},  # bool is not an int
    {"leave_type": "casual", "days": 1.5, "start_date": "2026-11-02"},   # not whole
    {"leave_type": "casual", "days": 0, "start_date": "2026-11-02"},
    {"leave_type": "casual", "days": 31, "start_date": "2026-11-02"},
    {"leave_type": "casual", "days": 1},                                 # missing date
])
def test_validate_args_rejects_bad_leave_args(args):
    _, _, reg = build_agent()
    assert validate_args(reg["submit_leave_request"].params, args)


def test_working_days_counting():
    assert T.working_days_between(date(2026, 10, 19), date(2026, 10, 23)) == 3   # Tue Wed Thu
    assert T.working_days_between(date(2026, 10, 19), date(2026, 10, 20)) == 0
    assert T.working_days_between(date(2026, 10, 23), date(2026, 10, 26)) == 0   # Fri -> Mon: weekend only


def _pre(reg, **kw):
    args = {"leave_type": "casual", "days": 1, "start_date": "2026-11-02", **kw}
    return reg["submit_leave_request"].precheck(args)


def test_notice_period_boundary():
    _, _, reg = build_agent()
    assert _pre(reg, start_date="2026-10-23") is None          # exactly 3 working days: allowed
    assert "working days" in _pre(reg, start_date="2026-10-22")  # 2: rejected


def test_sick_leave_is_exempt_from_notice_and_past_dates_are_rejected():
    _, _, reg = build_agent()
    assert _pre(reg, leave_type="sick", start_date="2026-10-20") is None
    assert "past" in _pre(reg, start_date="2026-10-01")
    assert "valid date" in _pre(reg, start_date="2026-13-45")
    assert "valid date" in _pre(reg, start_date="20261102")      # basic ISO form must not slip through


def test_allowance_is_cumulative_and_overlaps_are_rejected():
    agent, ledger, reg = build_agent()
    ledger.add("casual", 10, date(2026, 11, 2))
    assert "only 2" in _pre(reg, days=3, start_date="2026-12-01")
    assert _pre(reg, days=2, start_date="2026-12-01") is None
    assert "overlaps" in _pre(reg, days=1, start_date="2026-11-05")
