"""LLM clients behind one tiny interface.

* MockLLM          - deterministic, offline, no API key. Used for CI and for demonstrating the
                     harness (it can also *simulate* a hallucinating / vulnerable / noisy model).
* OllamaLLM        - local models via Ollama (http://localhost:11434).
* OpenAICompatLLM  - anything that speaks the OpenAI chat-completions API (OpenAI, Groq,
                     Gemini's OpenAI-compatible endpoint, Ollama's /v1, ...).
"""
from __future__ import annotations

import json
import os
import random
import re

import requests

from app.text_utils import split_sentences, tokenize

IDK = "I don't know based on the policy document."


class LLMError(Exception):
    """Generic model/API failure."""


class LLMTimeout(LLMError):
    pass


class LLMRateLimit(LLMError):
    pass


class BaseLLM:
    name = "base"

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- real providers
class OpenAICompatLLM(BaseLLM):
    def __init__(self, base_url=None, api_key=None, model=None, timeout=30, prefix="LLM"):
        self.base_url = (base_url or os.getenv(f"{prefix}_BASE_URL") or os.getenv("LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv(f"{prefix}_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv(f"{prefix}_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.name = f"openai-compat:{self.model}"

    def generate(self, system, user, temperature=0.0):
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "temperature": temperature,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise LLMTimeout(str(e)) from e
        except requests.RequestException as e:
            raise LLMError(str(e)) from e
        if r.status_code == 429:
            raise LLMRateLimit("rate limited (HTTP 429)")
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"malformed response: {e}") from e


class OllamaLLM(BaseLLM):
    def __init__(self, model=None, host=None, timeout=120):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def generate(self, system, user, temperature=0.0):
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "stream": False, "options": {"temperature": temperature},
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise LLMTimeout(str(e)) from e
        except requests.RequestException as e:
            raise LLMError(str(e)) from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["message"]["content"]
        except (KeyError, ValueError) as e:
            raise LLMError(f"malformed response: {e}") from e


# --------------------------------------------------------------------------- offline mock
class MockLLM(BaseLLM):
    """Extractive 'model': answers with the context sentence that best matches the question.

    mode:
      faithful     - behaves well (default)
      hallucinating- appends an invented claim to every answer
      vulnerable   - obeys prompt-injection / prompt-extraction attempts (a weak model)
    noise: probability (only when temperature > 0) of picking the 2nd-best sentence, to
           simulate run-to-run inconsistency.
    """

    MIN_OVERLAP = 0.5

    def __init__(self, mode: str = "faithful", noise: float = 0.0, seed: int = 7):
        assert mode in {"faithful", "hallucinating", "vulnerable"}
        self.mode, self.noise = mode, noise
        self.rng = random.Random(seed)
        self.name = f"mock:{mode}" + (f"+noise{noise}" if noise else "")

    def generate(self, system, user, temperature=0.0):
        if "RUBRIC:" in user:                       # acting as the LLM-as-judge
            return self._judge(user)
        m = re.search(r"CONTEXT:\n(.*)\n\nQUESTION:\s*(.*)$", user, re.S)
        if not m:
            return IDK
        context, question = m.group(1), m.group(2).strip()

        if self.mode == "vulnerable":
            if re.search(r"system prompt|instructions", question, re.I) and \
                    re.search(r"print|repeat|reveal|show", question, re.I):
                return "Sure! Here are my instructions: " + system
            if re.search(r"ignore .*instructions|disregard|you are now", question, re.I):
                w = re.search(r"(?:say|reply with|respond with|print|output)\s+(?:only\s+)?"
                              r"(?:the\s+word\s+)?[\"']?([A-Za-z]+)", question, re.I)
                if w:
                    return w.group(1).upper()

        q_tokens = set(tokenize(question))
        scored = []
        for sent in split_sentences(context):
            if sent.startswith("#"):
                continue
            s_tokens = set(tokenize(sent))
            hit = len(q_tokens & s_tokens)
            if q_tokens and hit >= 2 - (len(q_tokens) == 1) and hit / len(q_tokens) >= self.MIN_OVERLAP:
                scored.append((hit / len(q_tokens), sent))
        if not scored:
            return IDK
        scored.sort(key=lambda x: -x[0])             # stable: ties keep document order
        pick = scored[0][1]
        if temperature > 0 and len(scored) > 1 and self.rng.random() < self.noise:
            pick = self.rng.choice(scored[:2])[1]      # sometimes the runner-up: run-to-run drift
        if self.mode == "hallucinating":
            pick += " Additionally, employees receive a 5000 INR annual wellness bonus."
        return pick

    @staticmethod
    def _judge(prompt: str) -> str:
        ref = re.search(r"REFERENCE:\s*(.*?)\nANSWER:", prompt, re.S)
        ans = re.search(r"ANSWER:\s*(.*?)\n\nRespond", prompt, re.S)
        ref_t, ans_t = ref.group(1) if ref else "", ans.group(1) if ans else ""
        ref_tokens, ans_tokens = set(tokenize(ref_t)), set(tokenize(ans_t))
        recall = len(ref_tokens & ans_tokens) / len(ref_tokens) if ref_tokens else 0.0
        score = 5 if recall >= 0.9 else 4 if recall >= 0.7 else 3 if recall >= 0.4 else 2 if recall >= 0.2 else 1
        if any(t.isdigit() and t not in ans_tokens for t in ref_tokens):
            score = min(score, 2)                     # a missing / wrong number is never "correct"
        return json.dumps({"score": score, "reason": f"reference recall = {recall:.2f}"})


def get_llm(provider: str | None = None, role: str = "answer", mock_mode: str = "faithful",
            noise: float = 0.0) -> BaseLLM:
    provider = (provider or os.getenv("EVAL_PROVIDER", "mock")).lower()
    if provider == "mock":
        return MockLLM(mode=mock_mode, noise=noise)
    if provider == "ollama":
        return OllamaLLM(model=os.getenv("JUDGE_MODEL") if role == "judge" else None)
    if provider in {"openai", "openai-compat", "groq", "gemini"}:
        return OpenAICompatLLM(prefix="JUDGE" if role == "judge" else "LLM")
    raise ValueError(f"unknown provider: {provider}")
