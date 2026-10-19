"""Does the model give the same answer when asked the same thing several times?"""
from itertools import combinations

from app.text_utils import cosine_tokens, normalize


def consistency_score(answers: list[str]) -> dict:
    if len(answers) < 2:
        return {"mean_pairwise_cosine": 1.0, "agreement_rate": 1.0}
    pairs = list(combinations(answers, 2))
    cos = sum(cosine_tokens(a, b) for a, b in pairs) / len(pairs)
    norm = [normalize(a) for a in answers]
    modal = max(set(norm), key=norm.count)
    return {"mean_pairwise_cosine": round(cos, 3), "agreement_rate": round(norm.count(modal) / len(norm), 3)}
