"""Tiny shared text helpers (tokenising, normalising, sentence splitting)."""
import math
import re
from collections import Counter

STOPWORDS = set(
    """a an the is are was were be been am do does did how what which who whom whose when
    where why can could should would will shall may might must i me my we our you your he she
    it its they their them of in on at to for from by with about as into than that this these
    those and or but if then so not no any all each per many much there here also s t please
    tell give""".split()
)
# 'must' is a useful content word in policy text, so keep it out of the stop list.
STOPWORDS.discard("must")


def stem(token: str) -> str:
    """Very crude plural stripper: 'leaves' -> 'leave', 'days' -> 'day'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def normalize(text: str) -> str:
    """Lowercase, drop thousands separators (1,500 -> 1500), collapse whitespace."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def cosine_tokens(a: str, b: str) -> float:
    """Bag-of-words cosine similarity (a cheap, explainable stand-in for embeddings)."""
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb)
