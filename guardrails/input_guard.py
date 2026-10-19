"""Input guardrail: runs BEFORE the model sees the question."""
import re
from dataclasses import dataclass

MAX_LEN = 1000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|your\s+|the\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|prompt)",
    r"disregard\s+(your|the|all|any)",
    r"\bsystem\s+prompt\b",
    r"instructions\s+(were\s+you|you\s+were)\s+given",
    r"(reveal|repeat|print|show|output)\s+(me\s+)?(your|the)\s+(hidden\s+)?(instructions|prompt)",
    r"you\s+are\s+now\b",
    r"developer\s+mode|system\s+override|\bjailbreak\b|\bDAN\b",
    r"no\s+restrictions|without\s+restrictions",
    r"role-?play|pretend\s+(to\s+be|you\s+are)",
]
UNSAFE_PATTERNS = [
    r"\bhack(ing)?\b|\bexploit\b|\bsteal\b|\bbypass\b",
    r"\b(make|build|create)\s+(a\s+)?(bomb|weapon|malware|virus)\b",
]


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"      # ok | empty | too_long | injection | unsafe
    reason: str = ""


def check_input(text) -> GuardResult:
    if not isinstance(text, str) or not text.strip():
        return GuardResult(False, "empty", "empty input")
    if len(text) > MAX_LEN:
        return GuardResult(False, "too_long", f"input longer than {MAX_LEN} characters")
    for p in INJECTION_PATTERNS:
        if re.search(p, text, re.I):
            return GuardResult(False, "injection", f"matched /{p}/")
    for p in UNSAFE_PATTERNS:
        if re.search(p, text, re.I):
            return GuardResult(False, "unsafe", f"matched /{p}/")
    return GuardResult(True)
