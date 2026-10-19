"""Failure scenarios: the model/API misbehaves. The bot must degrade gracefully and stay safe."""
import pytest
import requests

from app.llm_client import LLMError, LLMRateLimit, LLMTimeout, OpenAICompatLLM
from app.policy_bot import MSG_BLOCKED_OUTPUT, PolicyBot
from tests.conftest import SequenceLLM

GOOD = "Casual leave: every employee gets 12 days of casual leave per year."


def test_transient_timeout_is_retried():
    llm = SequenceLLM(LLMTimeout("slow"), GOOD)
    r = PolicyBot(llm, max_retries=1).ask("How many casual leaves do I get?")
    assert r.status == "answered" and llm.calls == 2


def test_persistent_timeout_returns_fallback_not_exception():
    r = PolicyBot(SequenceLLM(LLMTimeout("slow")), max_retries=2).ask("How many casual leaves do I get?")
    assert r.status == "error" and "temporarily unavailable" in r.answer


def test_rate_limit_is_retried_then_reported():
    llm = SequenceLLM(LLMRateLimit("429"))
    r = PolicyBot(llm, max_retries=1).ask("How many casual leaves do I get?")
    assert r.status == "error" and llm.calls == 2


@pytest.mark.parametrize("bad_output", ["", "   ", None, 42])
def test_empty_or_malformed_model_output(bad_output):
    r = PolicyBot(SequenceLLM(bad_output)).ask("How many casual leaves do I get?")
    assert r.status == "error"


def test_non_retryable_error_is_not_retried():
    llm = SequenceLLM(LLMError("bad request"))
    r = PolicyBot(llm, max_retries=3).ask("How many casual leaves do I get?")
    assert r.status == "error" and llm.calls == 1


def test_output_guard_catches_a_model_that_leaks():
    """The model misbehaves on its own (no attack input needed) - the output guard is the safety net."""
    leaky = SequenceLLM("Sure, the key is ACME-KEY-93X7-QZ41.")
    r = PolicyBot(leaky).ask("How many casual leaves do I get?")
    assert r.status == "blocked_output" and r.answer == MSG_BLOCKED_OUTPUT


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.text = code, body or {}, "x"

    def json(self):
        return self._b


def test_http_429_maps_to_rate_limit(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(429))
    with pytest.raises(LLMRateLimit):
        OpenAICompatLLM(base_url="http://x", api_key="k", model="m").generate("s", "u")


def test_http_timeout_maps_to_llm_timeout(monkeypatch):
    def boom(*a, **k):
        raise requests.Timeout("t")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMTimeout):
        OpenAICompatLLM(base_url="http://x", api_key="k", model="m").generate("s", "u")


def test_malformed_http_body_maps_to_llm_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, {"unexpected": 1}))
    with pytest.raises(LLMError):
        OpenAICompatLLM(base_url="http://x", api_key="k", model="m").generate("s", "u")
