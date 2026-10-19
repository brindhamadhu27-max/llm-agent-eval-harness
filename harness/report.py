"""Writes latest.json, appends history.csv (quality over time) and renders report.html."""
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.config import check_thresholds

HISTORY_FIELDS = ["timestamp", "label", "correctness_pass_rate", "judge_mean", "faithfulness",
                  "consistency", "refusal_rate", "attack_resistance", "edge_graceful_rate", "gate"]


def save(result: dict, out_dir: Path, label: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = check_thresholds(result["metrics"])
    passed = all(g["ok"] for g in gates)
    (out_dir / "latest.json").write_text(
        json.dumps({"label": label, "gates": gates, **result}, indent=2), encoding="utf-8")

    hist = out_dir / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "label": label, **result["metrics"], "gate": "PASS" if passed else "FAIL"})
    rows = list(csv.DictReader(hist.open(encoding="utf-8")))
    (out_dir / "report.html").write_text(render_html(result, gates, rows, label), encoding="utf-8")
    return passed


def _trend_svg(rows: list[dict], keys=("correctness_pass_rate", "attack_resistance", "faithfulness")) -> str:
    rows = rows[-20:]
    if len(rows) < 2:
        return "<p><em>Run the evaluation a few more times to see a trend.</em></p>"
    w, h, pad = 560, 160, 28
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    out.append(f'<rect x="{pad}" y="8" width="{w-pad-8}" height="{h-pad-8}" fill="none" stroke="#999"/>')
    for k, col in zip(keys, colors):
        pts = []
        for i, r in enumerate(rows):
            x = pad + i * (w - pad - 8) / (len(rows) - 1)
            y = 8 + (1 - float(r[k])) * (h - pad - 8)
            pts.append(f"{x:.1f},{y:.1f}")
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{" ".join(pts)}"/>')
    legend = " ".join(f'<span style="color:{c}">&#9632; {k}</span>' for k, c in zip(keys, colors))
    return "".join(out) + f"</svg><div style='font-size:12px'>{legend} (scale 0 to 1)</div>"


def render_html(result, gates, rows, label) -> str:
    e = html.escape
    g_rows = "".join(
        f"<tr><td>{e(g['metric'])}</td><td>{g['value']}</td><td>&ge; {g['threshold']}</td>"
        f"<td style='color:{'#16a34a' if g['ok'] else '#dc2626'}'>{'PASS' if g['ok'] else 'FAIL'}</td></tr>"
        for g in gates)
    c_rows = "".join(
        f"<tr><td>{e(c['suite'])}</td><td>{e(c['id'])}</td><td>{e(c['question'])}</td>"
        f"<td>{e(c['status'])}</td><td>{e(c['answer'][:110])}</td>"
        f"<td style='color:{'#16a34a' if c['passed'] else '#dc2626'}'>{'PASS' if c['passed'] else 'FAIL'}</td></tr>"
        for c in result["cases"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM eval report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}}th{{background:#f3f4f6}}</style>
</head><body><h1>LLM evaluation report</h1><p><b>Run:</b> {e(label)}</p>
<h2>Quality gates</h2><table><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Gate</th></tr>{g_rows}</table>
<h2>Trend over runs</h2>{_trend_svg(rows)}
<h2>Every test case</h2><table><tr><th>Suite</th><th>ID</th><th>Question</th><th>Status</th><th>Answer</th><th>Result</th></tr>{c_rows}</table>
</body></html>"""
