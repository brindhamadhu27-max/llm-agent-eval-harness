# Interview guide - the 15-minute walkthrough

Be honest about one thing up front: the default **mock model** exists so tests are deterministic
and free in CI. The real value comes from running the same harness against a **real model**
(Ollama or any OpenAI-compatible API) and reporting what you actually found. Do that before the interview.

## Before the interview (checklist)
- [ ] Read every file once; you must be able to explain any line.
- [ ] Run `pytest -q` and `python run_eval.py` on a fresh clone.
- [ ] Run against a real model, with guardrails ON and OFF. Put the numbers in the README "Results" table.
- [ ] Write down ONE real failure you found (e.g. a real model leaked the key, hallucinated a number, refused a legitimate question, or the judge gave an odd score) and how you fixed it.
- [ ] Keep `reports/report.html` open and a backup screen recording of a run.

## Timeline
| Min | Show | Say |
|---|---|---|
| 0-2 | README architecture diagram | "A policy Q&A bot with planted fake secrets, and a harness that measures quality and security, gated in CI." |
| 2-5 | `app/policy_bot.py`, `guardrails/`, `datasets/` | The flow, why guards run on both sides of the model, how datasets are organised (golden / unanswerable / adversarial / edge). |
| 5-10 | Terminal: `pytest -q`, then `python run_eval.py`, open `report.html` | Each metric and how it is computed. Then run `--mock-mode vulnerable --no-guardrails` to show a failing gate, then with guardrails to show it pass. |
| 10-13 | Your real-model finding | Symptom -> root cause -> fix -> test that now prevents regression. |
| 13-15 | README "Limitations" | Judge bias, lexical similarity limits, small datasets, what you'd do next. |

## Mapping to their questions
**Area 2 - LLM testing/evaluation**
| They ask | Where to point |
|---|---|
| How did you build the framework? | `harness/runner.py`, `evaluators/` |
| Test agentic skills and validate outputs | `PolicyBot.ask` pipeline; each stage (retrieve, generate, guard) has tests. Extension idea: add a tool and test tool selection. |
| Accuracy, relevance, consistency, hallucination | `correctness.py`, `judge.py`, `consistency.py`, `faithfulness.py` |
| How do you decide a response is correct? | Hard fact check first (deterministic), judge for nuance, human spot-checks to calibrate |
| Guardrails | `guardrails/`, `tests/test_guardrails.py`, `test_guardrails_are_what_protects_us` |
| Injection, unsafe, leakage | `datasets/adversarial.json`, `tests/test_safety.py`, `evaluators/safety.py` |
| Metrics and monitoring over time | `harness/config.py` gates, `reports/history.csv` trend, weekly CI schedule |

**Area 1 - agent testing (if asked):** treat the bot as an agent with skills: retrieve -> answer -> guard.
Positive/negative/edge/failure are separate test files; guardrail actions are asserted via `status`.

## Likely follow-up questions
1. **Why not trust an LLM judge blindly?** Verbosity/position/self-preference bias and non-determinism. Mitigate: temperature 0, reference answers, a narrow rubric, a different judge model, and calibrate against human-labelled samples (report agreement).
2. **How do you test non-deterministic outputs?** Assert properties not strings (facts present, grounded, safe), repeat runs, use thresholds and confidence intervals, fix temperature/seed where possible.
3. **How do you detect model drift?** Scheduled runs of a fixed dataset, trend in `history.csv`, alert when a gate drops; pin model versions; keep a canary set.
4. **Precision vs recall of guardrails?** Blocking everything is "safe" but useless: the tests check legitimate questions are allowed (false-positive tests).
5. **Why independent safety oracles?** So a bug in the guard cannot hide itself from the test.
6. **What are the limits of keyword-based faithfulness?** Misses paraphrase and subtle reasoning errors; next step is an NLI/entailment model or claim-level LLM verification.
7. **How would this scale?** Larger versioned datasets, sampling real traffic (with privacy review), human review queue for low-confidence cases, dashboards, per-category regressions.
8. **How do you handle prompt injection through retrieved documents?** Treat retrieved text as untrusted data, delimiters, output filtering, least-privilege tools; add indirect-injection cases to the dataset.
