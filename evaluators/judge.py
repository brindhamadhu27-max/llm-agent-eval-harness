"""LLM-as-judge with an explicit rubric and a defensive JSON parser.

Known weaknesses (say these in the interview): position/verbosity bias, self-preference
(use a different model than the one being judged), non-determinism.  Mitigations: temperature 0,
a reference answer, a narrow rubric, and spot-checking judge scores against human labels.
"""
import json
import re

PROMPT = """You are grading an answer from a company policy assistant.

RUBRIC:
5 = fully correct and complete; 4 = correct, minor omission or extra detail;
3 = partly correct; 2 = mostly wrong or off-topic; 1 = wrong, invented, or refuses wrongly.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def parse_judge_output(text: str):
    """Extract {'score': int, 'reason': str}; return None if unusable."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        score = int(data["score"])
    except (ValueError, KeyError, TypeError):
        return None
    if not 1 <= score <= 5:
        return None
    return {"score": score, "reason": str(data.get("reason", ""))}


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def score(self, question: str, reference: str, answer: str) -> dict:
        raw = self.llm.generate("You are a strict, concise grader.",
                                PROMPT.format(question=question, reference=reference, answer=answer), 0.0)
        parsed = parse_judge_output(raw)
        return parsed or {"score": 1, "reason": "unparseable judge output"}
