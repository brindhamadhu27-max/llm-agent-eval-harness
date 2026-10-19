"""Happy path: the bot answers answerable questions correctly, faithfully and well."""
import pytest

from evaluators.correctness import score_correctness
from evaluators.faithfulness import faithfulness_score
from tests.conftest import dataset

CASES = dataset("golden")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden_question(case, bot, judge):
    r = bot.ask(case["question"])
    assert r.status == "answered", r.answer
    assert score_correctness(r.answer, case)["passed"], f"missing {case['expected_keywords']} in: {r.answer}"
    assert faithfulness_score(r.answer, r.contexts)["score"] >= 0.8
    assert judge.score(case["question"], case["expected_answer"], r.answer)["score"] >= 4
