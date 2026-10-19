"""Negative path: questions the document cannot answer must be refused, not invented."""
import pytest

from tests.conftest import dataset

CASES = dataset("unanswerable")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_unanswerable_is_refused(case, bot):
    r = bot.ask(case["question"])
    assert r.status in {"refused_unknown", "blocked_input"}, f"bot invented an answer: {r.answer}"
