"""Consistency: same question, several runs -> (nearly) same answer."""
from app.llm_client import MockLLM
from app.policy_bot import PolicyBot
from evaluators.consistency import consistency_score

Q = "How many days of sick leave do I get?"


def _answers(llm, n=6):
    bot = PolicyBot(llm)
    return [bot.ask(Q, temperature=0.7).answer for _ in range(n)]


def test_stable_model_is_consistent():
    assert consistency_score(_answers(MockLLM()))["agreement_rate"] == 1.0


def test_metric_detects_an_unstable_model():
    s = consistency_score(_answers(MockLLM(noise=1.0)))
    assert s["agreement_rate"] < 1.0 or s["mean_pairwise_cosine"] < 1.0


def test_identical_answers_score_one():
    assert consistency_score(["a b c", "a b c"])["mean_pairwise_cosine"] == 1.0
