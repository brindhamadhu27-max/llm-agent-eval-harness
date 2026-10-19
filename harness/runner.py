"""Runs every dataset through the bot and aggregates metrics."""
import json
from pathlib import Path
from statistics import mean

from evaluators.consistency import consistency_score
from evaluators.correctness import score_correctness
from evaluators.faithfulness import faithfulness_score
from evaluators.safety import SAFE_STATUSES, attack_resisted

DATA = Path(__file__).resolve().parent.parent / "datasets"


def load(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def run_all(bot, judge, runs: int = 3, consistency_cases: int = 10) -> dict:
    cases, golden_rows = [], []

    for c in load("golden"):                                   # 1. correctness / quality
        r = bot.ask(c["question"])
        corr = score_correctness(r.answer, c)
        faith = faithfulness_score(r.answer, r.contexts)
        j = judge.score(c["question"], c["expected_answer"], r.answer)
        golden_rows.append((c, corr, faith, j))
        cases.append({"suite": "golden", "id": c["id"], "question": c["question"], "answer": r.answer,
                      "status": r.status, "passed": corr["passed"] and r.status == "answered",
                      "details": {"correctness": corr, "faithfulness": faith["score"], "judge": j}})

    cons = []                                                  # 2. consistency (repeat at temp 0.7)
    for c in load("golden")[:consistency_cases]:
        answers = [bot.ask(c["question"], temperature=0.7).answer for _ in range(runs)]
        cons.append(consistency_score(answers)["mean_pairwise_cosine"])

    unans = []                                                 # 3. hallucination resistance
    for c in load("unanswerable"):
        r = bot.ask(c["question"])
        ok = r.status in {"refused_unknown", "blocked_input"}
        unans.append(ok)
        cases.append({"suite": "unanswerable", "id": c["id"], "question": c["question"],
                      "answer": r.answer, "status": r.status, "passed": ok, "details": {}})

    attacks = []                                               # 4. security
    for c in load("adversarial"):
        r = bot.ask(c["question"])
        res = attack_resisted(c, r.answer, r.status)
        attacks.append(res["resisted"])
        cases.append({"suite": "adversarial", "id": c["id"], "question": c["question"][:120],
                      "answer": r.answer, "status": r.status, "passed": res["resisted"],
                      "details": {"type": c["type"], "leaked": res["leaked"], "detail": r.detail}})

    edge = []                                                  # 5. edge + failure handling
    for c in load("edge"):
        try:
            r = bot.ask(c["question"])
            ok = r.status in c["expect_status"]
            status, answer = r.status, r.answer
        except Exception as e:                                 # an exception IS a failure
            ok, status, answer = False, "exception", repr(e)
        edge.append(ok)
        cases.append({"suite": "edge", "id": c["id"], "question": c["question"][:60],
                      "answer": answer[:120], "status": status, "passed": ok,
                      "details": {"note": c["note"], "expected": c["expect_status"]}})

    metrics = {
        "correctness_pass_rate": mean(x[1]["passed"] for x in golden_rows),
        "judge_mean": mean(x[3]["score"] for x in golden_rows),
        "faithfulness": mean(x[2]["score"] for x in golden_rows),
        "consistency": mean(cons),
        "refusal_rate": mean(unans),
        "attack_resistance": mean(attacks),
        "edge_graceful_rate": mean(edge),
    }
    return {"metrics": {k: round(float(v), 3) for k, v in metrics.items()}, "cases": cases}
