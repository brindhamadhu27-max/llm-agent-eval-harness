"""End-to-end smoke test of the reporting pipeline and quality gates."""
import json

from app.llm_client import MockLLM
from app.policy_bot import PolicyBot
from evaluators.judge import LLMJudge
from harness.config import THRESHOLDS, check_thresholds
from harness.report import save
from harness.runner import run_all


def test_full_run_passes_all_gates_and_writes_reports(tmp_path):
    result = run_all(PolicyBot(MockLLM()), LLMJudge(MockLLM()), runs=2)
    assert set(result["metrics"]) == set(THRESHOLDS)
    assert all(g["ok"] for g in check_thresholds(result["metrics"]))
    assert save(result, tmp_path, "test") is True
    assert json.loads((tmp_path / "latest.json").read_text())["gates"]
    assert (tmp_path / "report.html").exists() and (tmp_path / "history.csv").exists()


def test_quality_gate_fails_for_broken_model(tmp_path):
    """The gate must actually fail when quality drops (otherwise CI protects nothing)."""
    result = run_all(PolicyBot(MockLLM(mode="hallucinating")), LLMJudge(MockLLM()), runs=2)
    assert save(result, tmp_path, "broken") is False


def test_unprotected_vulnerable_bot_fails_security_gate(tmp_path):
    result = run_all(PolicyBot(MockLLM(mode="vulnerable"), guardrails=False), LLMJudge(MockLLM()), runs=2)
    assert result["metrics"]["attack_resistance"] < 1.0
