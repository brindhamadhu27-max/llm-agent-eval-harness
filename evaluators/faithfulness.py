"""Hallucination check: is every sentence of the answer supported by the retrieved context?

Rule of thumb used here (simple on purpose, easy to explain and to replace with an NLI model):
a sentence is *supported* if >= 70% of its content tokens occur in the context AND every
number it mentions occurs in the context.  Refusals ("I don't know...") are supported.
"""
from app.text_utils import split_sentences, tokenize

SUPPORT_RATIO = 0.7


def faithfulness_score(answer: str, contexts: list[str]) -> dict:
    ctx_tokens = set(tokenize(" ".join(contexts)))
    sentences = split_sentences(answer)
    unsupported = []
    for sent in sentences:
        if "i don't know" in sent.lower():
            continue
        toks = tokenize(sent)
        if not toks:
            continue
        ratio = sum(t in ctx_tokens for t in toks) / len(toks)
        numbers_ok = all(t in ctx_tokens for t in toks if t.isdigit())
        if ratio < SUPPORT_RATIO or not numbers_ok:
            unsupported.append(sent)
    total = len(sentences) or 1
    return {"score": round(1 - len(unsupported) / total, 3), "unsupported": unsupported}
