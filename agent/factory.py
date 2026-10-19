"""One place that wires bot + ledger + tools + planner + agent (used by tests, eval and the CLI)."""
from __future__ import annotations

from datetime import date

from agent.agent import PolicyAgent
from agent.planner import LLMPlanner, RulePlanner
from agent.tools import LeaveLedger, build_registry
from app.llm_client import MockLLM, get_llm
from app.policy_bot import PolicyBot

FIXED_TODAY = date(2026, 10, 19)   # a Monday; fixed so working-day rules are deterministic in tests


def build_agent(llm=None, planner=None, guardrails: bool = True, today: date = FIXED_TODAY,
                ledger: LeaveLedger | None = None, provider: str = "mock"):
    """Returns (agent, ledger, registry). provider='mock' -> RulePlanner; else LLMPlanner on a real model."""
    llm = llm or (MockLLM() if provider == "mock" else get_llm(provider))
    bot = PolicyBot(llm, guardrails=guardrails)
    ledger = ledger or LeaveLedger()
    registry = build_registry(bot, ledger, today)
    if planner is None:
        planner = RulePlanner() if provider == "mock" else LLMPlanner(llm, registry)
    return PolicyAgent(planner, registry, guardrails=guardrails), ledger, registry
