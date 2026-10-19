"""agent_latest.json, agent_history.csv (quality over time) and agent_report.html."""
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.agent_runner import check_agent_thresholds

FIELDS = ["timestamp", "label", "task_success_rate", "tool_selection_accuracy", "argument_accuracy",
          "safe_action_rate", "leak_free_rate", "graceful_failure_rate", "gate"]


def save_agent(result: dict, out_dir: Path, label: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = check_agent_thresholds(result["metrics"])
    passed = all(g["ok"] for g in gates)
    (out_dir / "agent_latest.json").write_text(json.dumps({"label": label, "gates": gates, **result}, indent=2),
                                               encoding="utf-8")
    hist = out_dir / "agent_history.csv"
    new = not hist.exists()
    with hist.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label,
                    **result["metrics"], "gate": "PASS" if passed else "FAIL"})
    (out_dir / "agent_report.html").write_text(_html(result, gates, label), encoding="utf-8")
    return passed


def _html(result, gates, label) -> str:
    e = html.escape
    col = lambda ok: "#16a34a" if ok else "#dc2626"
    g = "".join(f"<tr><td>{e(x['metric'])}</td><td>{x['value']}</td><td>&ge; {x['threshold']}</td>"
                f"<td style='color:{col(x['ok'])}'>{'PASS' if x['ok'] else 'FAIL'}</td></tr>" for x in gates)
    k = "".join(f"<tr><td>{e(n)}</td><td>{v['passed']} / {v['total']}</td></tr>" for n, v in result["by_kind"].items())
    c = "".join(
        f"<tr><td>{e(x['id'])}</td><td>{e(x['kind'])}</td><td>{e(x['request'])}</td><td>{e(x['fault'])}</td>"
        f"<td>{e(x['expected_status'])}</td><td>{e(x['status'])}</td><td>{e(' &rarr; '.join(x['tools']) or '-')}</td>"
        f"<td>{e(x['answer'])}</td><td style='color:{col(x['passed'])}'>{'PASS' if x['passed'] else 'FAIL ' + e(','.join(x['failed_checks']))}</td></tr>"
        for x in result["cases"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Agent eval report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1200px}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}}th{{background:#f3f4f6}}</style></head><body>
<h1>Agent evaluation report</h1><p><b>Run:</b> {e(label)}</p>
<h2>Quality gates</h2><table><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Gate</th></tr>{g}</table>
<h2>Coverage by scenario type</h2><table><tr><th>Type</th><th>Passed</th></tr>{k}</table>
<h2>Every trajectory</h2><table><tr><th>ID</th><th>Type</th><th>Request</th><th>Injected fault</th><th>Expected</th><th>Actual</th><th>Tools attempted</th><th>Answer</th><th>Result</th></tr>{c}</table>
</body></html>"""
