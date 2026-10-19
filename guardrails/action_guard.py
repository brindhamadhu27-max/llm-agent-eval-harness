"""Action guardrail for the agent: runs BEFORE any tool executes.

The planner (an LLM in production) is NOT trusted. Every proposed action must pass:
  1. allow-list      - the tool must exist in the registry
  2. schema          - exact argument names, types, enums and ranges
  3. business rules  - the tool's own precheck (allowance, notice period, overlaps ...)
  4. loop guard      - the same call may not be repeated within one run
  5. blast radius    - at most `max_writes` state-changing actions per request
Confirmation of write actions (human-in-the-loop) is enforced by the agent after this check.
"""
from dataclasses import dataclass


@dataclass
class ActionDecision:
    allowed: bool
    category: str = "ok"   # ok | unknown_tool | bad_args | precheck | repeated | too_many_writes
    reason: str = ""


def validate_args(spec: dict, args) -> str | None:
    """Return an error string, or None if args satisfy the spec."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    unknown = set(args) - set(spec)
    if unknown:
        return f"unexpected argument(s): {', '.join(sorted(map(str, unknown)))}"
    for name, rule in spec.items():
        if name not in args:
            if rule.get("required", True):
                return f"missing required argument '{name}'"
            continue
        v, t = args[name], rule["type"]
        if t == "str" and not isinstance(v, str):
            return f"'{name}' must be a string"
        if t == "int" and (not isinstance(v, int) or isinstance(v, bool)):
            return f"'{name}' must be a whole number"
        if t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            return f"'{name}' must be a number"
        if "enum" in rule and v not in rule["enum"]:
            return f"'{name}' must be one of {rule['enum']}"
        if "min" in rule and v < rule["min"]:
            return f"'{name}' must be at least {rule['min']}"
        if "max" in rule and v > rule["max"]:
            return f"'{name}' must be at most {rule['max']}"
        if "max_len" in rule and len(v) > rule["max_len"]:
            return f"'{name}' is longer than {rule['max_len']} characters"
        if t == "str" and not v.strip():
            return f"'{name}' must not be empty"
    return None


def check_action(registry, tool_name, args, seen: set, writes_so_far: int,
                 max_writes: int = 1) -> ActionDecision:
    if not isinstance(tool_name, str) or tool_name not in registry:
        return ActionDecision(False, "unknown_tool", f"tool {tool_name!r} is not on the allow-list")
    tool = registry[tool_name]
    err = validate_args(tool.params, args)
    if err:
        return ActionDecision(False, "bad_args", err)
    if tool.precheck:
        err = tool.precheck(args)
        if err:
            return ActionDecision(False, "precheck", err)
    if action_key(tool_name, args) in seen:
        return ActionDecision(False, "repeated", "identical action already executed in this run")
    if tool.writes and writes_so_far >= max_writes:
        return ActionDecision(False, "too_many_writes", f"at most {max_writes} write action(s) per request")
    return ActionDecision(True)


def action_key(tool_name, args) -> str:
    import json
    return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
