#!/usr/bin/env python3
"""Run the full evaluation suite and write reports/.   Exit code 1 if any quality gate fails.

Examples
  python run_eval.py                                   # offline, deterministic (mock model)
  python run_eval.py --mock-mode vulnerable --no-guardrails   # watch the attacks succeed
  python run_eval.py --provider ollama                 # local model
  python run_eval.py --provider openai --judge-provider ollama
"""
import argparse
import sys
from pathlib import Path

from app.llm_client import get_llm
from app.policy_bot import PolicyBot
from evaluators.judge import LLMJudge
from harness.config import check_thresholds
from harness.report import save
from harness.runner import run_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None, help="mock | ollama | openai (default: $EVAL_PROVIDER or mock)")
    ap.add_argument("--judge-provider", default=None, help="provider for the LLM-as-judge (default: same)")
    ap.add_argument("--mock-mode", default="faithful", choices=["faithful", "hallucinating", "vulnerable"])
    ap.add_argument("--mock-noise", type=float, default=0.0, help="simulate inconsistent answers (0-1)")
    ap.add_argument("--no-guardrails", action="store_true", help="disable input/output guardrails")
    ap.add_argument("--runs", type=int, default=3, help="repeats per question for consistency")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    llm = get_llm(a.provider, mock_mode=a.mock_mode, noise=a.mock_noise)
    judge = LLMJudge(get_llm(a.judge_provider or a.provider, role="judge"))
    bot = PolicyBot(llm, guardrails=not a.no_guardrails,
                    retry_delay=0.0 if llm.name.startswith("mock") else 1.0)
    label = f"{llm.name} | guardrails={'off' if a.no_guardrails else 'on'}"
    print(f"Running evaluation: {label}")

    result = run_all(bot, judge, runs=a.runs)
    ok = save(result, Path(a.out), label)

    print(f"\n{'metric':<24}{'value':>8}{'threshold':>11}  gate")
    for g in check_thresholds(result["metrics"]):
        print(f"{g['metric']:<24}{g['value']:>8}{g['threshold']:>11}  {'PASS' if g['ok'] else 'FAIL'}")
    failed = [c for c in result["cases"] if not c["passed"]]
    for c in failed:
        print(f"  x {c['suite']}/{c['id']}: status={c['status']} | {c['answer'][:70]}")
    print(f"\nReport: {Path(a.out) / 'report.html'}   ->  {'ALL GATES PASSED' if ok else 'QUALITY GATE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
