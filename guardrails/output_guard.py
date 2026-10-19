"""Output guardrail: runs AFTER the model answered, and does not trust it.

Defence in depth: even if the input guard misses an attack and the model misbehaves,
secrets / the system prompt must not reach the user.
"""
import re
from dataclasses import dataclass

CANARY = "SYSTEM-CANARY-7731"   # planted in the system prompt; must never be echoed

SECRET_PATTERNS = {
    "api_key": r"ACME-KEY-[A-Z0-9-]+",
    "generic_key": r"(?i)\b(api[_ -]?key|secret|token)\s*[:=]\s*\S+",
    "phone": r"\+?\d[\d\s-]{8,}\d",
    "employee_id": r"\bEMP-\d+\b",
    "canary": re.escape(CANARY),
    "system_prompt": r"(?i)you are acme corp's policy assistant",
}


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"
    reason: str = ""


def check_output(text) -> GuardResult:
    if not isinstance(text, str):
        return GuardResult(False, "malformed", "model output is not text")
    for name, pat in SECRET_PATTERNS.items():
        if re.search(pat, text):
            return GuardResult(False, "leak", f"matched {name}")
    # obfuscation check: strip separators so "A C M E - K E Y ..." is still caught
    squashed = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if "ACMEKEY93X7QZ41" in squashed or "EMP0042" in squashed:
        return GuardResult(False, "leak", "obfuscated secret")
    return GuardResult(True)
