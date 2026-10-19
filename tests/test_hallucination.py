"""Do our evaluators actually detect hallucination? (A test suite must be able to fail.)"""
from app.llm_client import MockLLM
from app.policy_bot import PolicyBot
from evaluators.faithfulness import faithfulness_score


def test_faithfulness_detects_invented_claim():
    bot = PolicyBot(MockLLM(mode="hallucinating"))
    r = bot.ask("How many casual leaves do I get in a year?")
    f = faithfulness_score(r.answer, r.contexts)
    assert f["score"] < 1.0 and any("5000" in s for s in f["unsupported"])


def test_faithful_model_scores_full_marks():
    r = PolicyBot(MockLLM()).ask("How many casual leaves do I get in a year?")
    assert faithfulness_score(r.answer, r.contexts)["score"] == 1.0


def test_invented_number_is_flagged_even_if_words_overlap():
    ctx = ["Casual leave: every employee gets 12 days of casual leave per year."]
    assert faithfulness_score("Every employee gets 15 days of casual leave per year.", ctx)["score"] == 0.0
