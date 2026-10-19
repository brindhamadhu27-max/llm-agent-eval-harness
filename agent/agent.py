"""PolicyAgent: plan -> guard -> (confirm) -> execute -> observe, with a hard step budget.

Safety properties (each one is asserted by tests/test_agent_*.py):
  * input guard first; a blocked sub-step halts the WHOLE run (fail closed)
  * every proposed action passes the action guard before it can run
  * write actions never run without explicit human confirmation; a confirmer that crashes = "no"
  * read tools are retried once on timeout; WRITE tools are never retried (no double-submit)
  * tool/planner crashes become a status, never an exception, and never leak raw error text
  * tool output and the final answer pass the output guard (secrets are never shown or traced)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.llm_client import LLMError
from app.policy_bot import MSG_BLOCKED_INPUT, MSG_BLOCKED_OUTPUT, MSG_ERROR
from agent.planner import BasePlanner, PlannerError
from agent.tools import ToolRegistry, ToolResult
from guardrails.action_guard import action_key, check_action
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output

MSG_MAX_STEPS = "Sorry, I couldn't finish that request within the allowed number of steps."
MSG_CANCELLED = "Okay, I have not submitted anything."
REDACTED = "[REDACTED]"
# completed | needs_confirmation | cancelled | needs_info | refused |
# blocked_input | blocked_action | blocked_output | error | max_steps


@dataclass
class AgentResult:
    request: str
    status: str
    answer: str
    trace: list = field(default_factory=list)
    pending: dict | None = None
    detail: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"request": self.request, "status": self.status, "answer": self.answer,
                "trace": self.trace, "pending": self.pending, "detail": self.detail}


class PolicyAgent:
    def __init__(self, planner: BasePlanner, registry: ToolRegistry, guardrails: bool = True,
                 max_steps: int = 5, max_writes: int = 1):
        self.planner, self.registry, self.guardrails = planner, registry, guardrails
        self.max_steps, self.max_writes = max_steps, max_writes

    def run(self, request, confirm=None) -> AgentResult:
        t0 = time.perf_counter()
        trace: list[dict] = []
        obs: list[dict] = []

        def done(status, answer, **kw) -> AgentResult:
            return AgentResult(request if isinstance(request, str) else repr(request), status, answer,
                               trace, latency_ms=(time.perf_counter() - t0) * 1000, **kw)

        if self.guardrails:
            g = check_input(request)
            if not g.allowed:
                return done("blocked_input", MSG_BLOCKED_INPUT, detail=f"{g.category}: {g.reason}")
        if not isinstance(request, str):
            return done("error", MSG_ERROR, detail="request is not a string")

        seen: set[str] = set()
        writes = 0
        for step in range(1, self.max_steps + 1):
            try:
                act = self.planner.next_action(request, obs)
            except (PlannerError, LLMError) as e:
                return done("error", MSG_ERROR, detail=f"planner: {type(e).__name__}: {e}")
            except Exception as e:                      # a planner bug must not crash the agent
                return done("error", MSG_ERROR, detail=f"planner crashed: {type(e).__name__}")

            if act.kind != "tool":                      # ---- terminal planner outputs
                body = "\n".join(o["message"] for o in obs)
                text = "\n".join(x for x in (body, act.message) if x) or "No action was needed."
                status = {"final": "completed", "clarify": "needs_info", "refuse": "refused"}.get(act.kind)
                if status is None:
                    return done("error", MSG_ERROR, detail=f"unknown action kind {act.kind!r}")
                if self.guardrails and not check_output(text).allowed:
                    trace.append({"step": step, "kind": act.kind, "message": REDACTED})
                    return done("blocked_output", MSG_BLOCKED_OUTPUT, detail="final answer failed output guard")
                trace.append({"step": step, "kind": act.kind, "message": text[:200]})
                return done(status, text)

            entry = {"step": step, "kind": "tool", "action": {"tool": act.tool, "args": act.args},
                     "decision": "allowed", "executed": False}
            trace.append(entry)

            if self.guardrails:                         # ---- action guard
                d = check_action(self.registry, act.tool, act.args, seen, writes, self.max_writes)
                if not d.allowed:
                    entry["decision"] = f"blocked:{d.category}"
                    entry["message"] = d.reason
                    return done("blocked_action", f"I can't do that: {d.reason}.", detail=f"{d.category}: {d.reason}")
            if act.tool not in self.registry:           # only reachable with guardrails off
                return done("error", MSG_ERROR, detail=f"unknown tool {act.tool!r}")
            tool = self.registry[act.tool]

            if tool.writes and self.guardrails:         # ---- human-in-the-loop for writes
                if confirm is None:
                    entry["decision"] = "pending_confirmation"
                    return done("needs_confirmation",
                                f"Please confirm: {act.tool} {act.args}. Nothing has been submitted yet.",
                                pending={"tool": act.tool, "args": act.args})
                try:
                    approved = bool(confirm({"tool": act.tool, "args": act.args}))
                except Exception:                       # fail closed
                    approved = False
                if not approved:
                    entry["decision"] = "declined"
                    return done("cancelled", MSG_CANCELLED)

            seen.add(action_key(act.tool, act.args))
            writes += tool.writes
            result = self._execute(tool, act.args)      # ---- execute (retry reads only)
            if result is None:
                entry["message"] = "tool failed"
                return done("error", MSG_ERROR, detail=f"tool {act.tool} failed")
            entry["executed"], entry["ok"] = True, result.ok

            if self.guardrails and not check_output(result.message).allowed:
                entry["message"] = REDACTED
                return done("blocked_output", MSG_BLOCKED_OUTPUT, detail=f"tool {act.tool} output failed output guard")
            entry["message"] = result.message[:200]
            term = result.data.get("terminal_status")
            if term:                                    # a blocked sub-step halts the whole run
                return done(term, result.message, detail=f"{act.tool} was blocked by its own guardrail")
            obs.append({"tool": act.tool, "args": act.args, "message": result.message})

        return done("max_steps", MSG_MAX_STEPS)

    def _execute(self, tool, args) -> ToolResult | None:
        attempts = 1 if tool.writes else 2              # never retry a write: could double-submit
        for i in range(attempts):
            try:
                return tool.handler(args)
            except TimeoutError:
                if i == attempts - 1:
                    return None
            except Exception:
                return None
        return None
