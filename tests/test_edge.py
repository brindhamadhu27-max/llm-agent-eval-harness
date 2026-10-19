"""Edge cases: odd input must be handled gracefully (no crash, acceptable status)."""
import pytest

from tests.conftest import dataset

CASES = dataset("edge")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_edge_case(case, bot):
    r = bot.ask(case["question"])
    assert r.status in case["expect_status"], f"{case['note']}: got {r.status}"


def test_non_string_input_does_not_crash(bot):
    assert bot.ask(None).status in {"blocked_input", "error"}
    assert bot.ask(12345).status in {"blocked_input", "error"}
