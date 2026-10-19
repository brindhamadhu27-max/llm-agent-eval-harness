"""Is the answer correct?  Three signals, from strict to soft."""
from app.text_utils import cosine_tokens, normalize, tokenize


def keywords_present(answer: str, keywords: list[str]) -> bool:
    """Hard check: every expected fact must appear (numbers, units, key phrases)."""
    a = normalize(answer)
    return all(normalize(k) in a for k in keywords)


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def score_correctness(answer: str, case: dict) -> dict:
    kw = keywords_present(answer, case["expected_keywords"])
    ref = case.get("expected_answer", "")
    return {
        "passed": kw,                                  # pass/fail is driven by the hard check
        "keywords_ok": kw,
        "f1": round(token_f1(answer, ref), 3),
        "cosine": round(cosine_tokens(answer, ref), 3),
    }
