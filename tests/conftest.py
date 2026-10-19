"""Shared fixtures.

Default provider is the offline mock so tests are free and deterministic.
Point at a real model with:  EVAL_PROVIDER=ollama pytest   (or openai - see README)
"""
import os

import pytest

from app.llm_client import MockLLM, get_llm
from app.policy_bot import PolicyBot
from evaluators.judge import LLMJudge
from harness.runner import load

REAL = os.getenv("EVAL_PROVIDER", "mock").lower() != "mock"


@pytest.fixture
def llm():
    return get_llm()


@pytest.fixture
def bot(llm):
    return PolicyBot(llm, guardrails=True)


@pytest.fixture
def judge():
    return LLMJudge(get_llm(os.getenv("JUDGE_PROVIDER"), role="judge"))


@pytest.fixture
def weak_model():
    """A model that obeys attacks. With the mock we simulate one; with a real provider use it as-is."""
    return get_llm() if REAL else MockLLM(mode="vulnerable")


def dataset(name):
    return load(name)


class SequenceLLM:
    """Test double: replays a scripted list of outputs/exceptions - for failure-scenario tests."""
    name = "stub"

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def generate(self, system, user, temperature=0.0):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step
