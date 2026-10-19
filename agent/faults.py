"""Fault injection: misbehaving planners and flaky tools, used to PROVE the guardrails work.
A well-behaved RulePlanner can never exercise the guards, so tests swap in these bad actors."""
from __future__ import annotations

from agent.planner import Action, BasePlanner


class GarbagePlanner(BasePlanner):
    def next_action(self, request, observations):
        from agent.planner import parse_action
        return parse_action("lol I'm not JSON")


class ScriptedPlanner(BasePlanner):
    """Replays fixed actions; after the script ends it returns `final`."""

    def __init__(self, *actions: Action):
        self.actions = list(actions)

    def next_action(self, request, observations):
        i = len(observations)
        return self.actions[i] if i < len(self.actions) else Action("final")


class LoopPlanner(BasePlanner):
    """Proposes the same call forever (runaway agent). Ignores observations on purpose."""

    def next_action(self, request, observations):
        return Action("tool", "policy_qa", {"question": "How many casual leaves do I get?"})


class AlwaysActionPlanner(BasePlanner):
    """Proposes a fixed list of actions regardless of what the guard did (a pushy model)."""

    def __init__(self, *actions: Action):
        self.actions, self.i = list(actions), 0

    def next_action(self, request, observations):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


def make_flaky(registry, tool_name: str, exc: Exception, times: int | None = 1):
    """Wrap a tool so it raises `exc` the first `times` calls (None = always). Returns call counter."""
    tool = registry[tool_name]
    real, state = tool.handler, {"calls": 0}

    def flaky(args):
        state["calls"] += 1
        if times is None or state["calls"] <= times:
            raise exc
        return real(args)

    tool.handler = flaky
    return state


def make_leaky(registry, tool_name: str, text: str):
    """Make a tool return `text` (e.g. a secret) as its message: tests the AGENT's own output guard,
    independent of the PolicyBot's guard that normally masks it."""
    from agent.tools import ToolResult
    registry[tool_name].handler = lambda args: ToolResult(True, text, {})
