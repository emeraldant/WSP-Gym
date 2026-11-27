# R4F — backdoor-witness, final state after two independent reviews

This report closes round 4. It replaces the numbers in `R4E.md`, which was written before the decoy design existed.
The task itself is in `discovery/wspnest/backdoor-witness/` (a mirror of the commit that would be the pull request);
its own record is `HEADROOM.md` in that folder.

## What exists
- **The task.** A model reads the weights of a small ReLU network (815-2,139 weights) and one line of
  specification, and answers with a witness (an input on which the network disagrees with the specification) or
  "clean". Networks come in matched poisoned/clean pairs. A forward pass scores a witness.
- **Frozen, certified data.** 450 networks in four files: plain design 100 val + 200 test, decoy design 50 val + 100
  test. Each passed a mixed-integer check over all its inputs: a clean twin never disagrees with its specification
  and a poisoned twin disagrees only inside its trigger. The data files did not change in either review.
- **One commit**, `de6e6b7`, on top of WSPNest pull request #1 (`wspnest-quality`, a3b4044). It is stored here as
  `discovery/wspnest/backdoor-witness.bundle`; the draft pull-request text is
  `discovery/wspnest/PR_backdoor-witness.md`.

## The two reviews
After the task was first declared finished, a separate agent that had not seen it being built audited the commit:
the verifier, label leaks, every number in the record, and the suite's rules. Its verdict: the certificates, the
data and almost every number hold, with one real verifier bug and a list of smaller discrepancies. A second
reviewer then checked the fixes and found one new bug in a check the fixes had added, plus wording and robustness
points. Everything both found is fixed or disclosed below. Their briefs and reports are in
`checkpoints/round_4/` (`audit_brief.md`, `audit_report.md`, `audit_brief_2.md`, `audit_report_2.md`). The first
auditor's checks are kept as `experiments/scripts/R4F_audit_*.py`; its certifier and its directed search were run
again here on all 450 networks. Outputs are in `experiments/outputs/R4F/audit/`.

### What the first audit confirmed
- **The certificates.** Its own checker, written from the instruction text with its own encoding and a margin ten
  times tighter, certifies all 450 networks and finds the backdoor in all 225 poisoned twins when the trigger is not excluded.
- **No error outside a trigger.** Its directed search: 0 errors in 3.5 billion forward passes over all 450 networks in the auditor's run, and 0 again in 0.9 billion with the committed script at its default settings; nothing came within the 0.001 margin (the closest input was 0.0012 from the threshold).
- **The certifier itself**, on small trained dense networks against exhaustive enumeration: 148 of 148, and 88 of 88
  with the trigger not excluded; no network with a real error was certified.
- **The data.** Weights parsed back from the instruction text equal the stored weights for all 450 networks; all 225
  reference witnesses are disagreements inside the trigger under its own forward pass.
- **The model results.** Every count and interval in both model tables, apart from the two items below.

### What the first audit found, and what was done
| Finding | Effect | Action |
|---|---|---|
| The answer parser took the last JSON object anywhere in a reply, so a refusal that quoted `{"clean": true}` scored as "clean" | One stored score was wrong (Muse Spark Open, `test_decoy` network 97). Muse's row was 29/50 clean and 0.30; it is 28/50 and 0.29 | Fixed. The answer is now the object that ends the reply, on a line of its own. Both parsers were run on all 695 stored replies (`R4F_record_facts.py`): they read 2 differently and one score changes. `selfcheck.py` now sends a format-quoting reply to every network and requires 0 |
| GPT-6.2 Sol's 0.85 depends on the token limit | 6 of its 100 calls ran out of tokens. On the 94 that answered it scores 0.90 | Disclosed in the record, the card and the pull-request text |
| The new test file made one of the suite's own tests skip silently | "14 passed, 1 skipped" hid a test that was no longer running | Fixed. The tests now load the task's modules without leaving them behind: none skipped |
| Task names gave the label away (even number = poisoned) and were too long for Harbor, which cuts trial names at 32 characters | No reported run saw a name. A Harbor run would have collapsed onto one or two reward keys, and an agent that saw its task's name could have read the label | Fixed. Tasks are numbered by pair with the twin order set by a hash; names are at most 32 characters; both are tested |
| The harness could not report the two shares or the pair score | `wspnest run` groups by a `clean` key the verifier did not write | Fixed. The verifier writes it; `pair_score.py` computes the pair score from a job's results |
| "More than 0.99 at 10^9" for uniform testing | It is 0.98 of plain triggers and 0.99 of decoy triggers | Corrected |
| The adoption pilot's "0.00" was one recipe; the rule said "10 networks" and the run covered 20 | The twelve recipes score at most 0.10 on the pilot; the first 10 networks alone also pass the rule | Disclosed |
| Weights-only detection is stronger than reported | A second statistic reaches 90% accuracy on decoy; a 25-statistic classifier 92-98.5% | Table added |
| Size range "1,000-2,100 weights" | It is 815-2,139 | Corrected |
| "No code" is not enforced in the container | An agent that ran a search would score about 0.9 | Disclosed; same convention as the suite's `wsp-gym` direct variant |
| Wording: "exact certificate", "no label matching", "every number can be re-run", two sentences of the instruction | Overstated or unclear | Reworded, except the instruction, which is unchanged because every model run used it; listed as a limit and raised with the maintainers |
| Smaller items: "wrong input" counts that included abstentions, stale run-time notes, an edit to the suite's shared `.gitignore`, training data building each pair twice | - | Fixed |

### What the second review found, and what was done
| Finding | Effect | Action |
|---|---|---|
| The new name check in `selfcheck.py` judged a short run of pairs by its share of poisoned-first twins | `wspnest check --opt limit=24` failed and would have overwritten `quality.json` with a fail, although nothing was wrong | Fixed. The check now tests the naming rule on whole splits; a test runs it at the limits that failed |
| Two sentences called the detector baseline "the strongest" and said no in-protocol script beats always-"clean" | On `test_decoy` that holds (best of 286 detector and recipe pairs: 0.47). On `test` 5 of 286 reach 0.505-0.515 | Corrected; all 286 pairs are now computed by a committed script and the best is on the card |
| The strict parser has two edges | A refusal whose last line is `{"clean": true}` still scores on a clean twin; `Answer: {"clean": true}` on the last line no longer scores. No stored reply is affected | Disclosed in the record, with tests that pin both behaviours. The rule was not loosened |
| Parser robustness | A deeply nested answer file raised an error; a very large one could run past the verifier's time limit | Fixed: such files score 0, quickly, and the verifier always writes its details |
| Four figures had no script | - | `R4F_record_facts.py` prints them |
| `pair_score.py` silently left out trials it could not read | Its numbers could differ from the harness's mean | It now reports incomplete pairs and rewards without details |
| `limit` with an odd number left gaps in the task numbers and always kept the poisoned twin; tasks were written poisoned twin first | No effect on a reported number | `limit` now keeps tasks 0 to limit-1, written in that order |
| Leftover wording: "exact", a paraphrase of the suite's bar in quotation marks, "re-run on all 450" | - | Corrected |
| Maintainer-facing gaps: no row in the suite's README table; "none skipped" needs scipy | - | Row added; stated in the pull-request text |

### Format check against the suite's contribution guide
Done after the two reviews, against `CONTRIBUTING.md`, the task template, the card schema, the website's merge
requirements and the three other task pull requests open upstream (which had not changed since the task was built).
The folder has the same shape as the maintainer's own CPU task. Three things were brought in line:
- The version is now 0.1.0, as in the template and in all three other tasks in review (it was 1.0.0).
- The canary is now also in the result files that describe test items (`baselines_*.json`, `data/build_log.json`)
  and in the record. The frozen test files are unchanged.
- The record credits the earlier work the task builds on, which the website asks for: TrojAI, Goldwasser et al. on
  planted backdoors, Tjeng et al. on mixed-integer verification, and the weight-space programming post.
The pull-request text now walks through the guide's eight steps.

## Checks that pass on the committed files
| Check | Result |
|---|---|
| `python -m wspnest check backdoor-witness` (all 450 networks, through `selfcheck.py`) | pass: oracle 1.0; missing, empty, other-twin and format-quoting answers all 0.0; deterministic; canary everywhere; nothing leaked; names fit and carry no label |
| `python -m wspnest validate` | ok |
| `pytest tests` in the WSPNest checkout, scipy installed | 18 passed, none skipped (13 tests are this task's) |
| `verify_data.py` on the four files | ok: twins matched, oracle answers verified, prompt round trip exact, no sampled error |
| `check_docker.py`, 20 generated tasks in real containers (10 plain, 10 decoy) | oracle 1.0, no answer 0.0, deterministic |
| A clean checkout restored from the bundle | `validate` ok; 18 tests pass, none skipped |
| Independent certifier on all 450 networks (`R4F_audit_milp.py`) | all 450 certified; the backdoor found in all 225 poisoned twins when the trigger is not excluded (3.3 CPU-hours) |
| Directed search outside the trigger on all 450 networks | 0 errors in 3.5 billion forward passes (auditor's run) and in 0.9 billion (re-run) |
| Certifier against exhaustive enumeration | 292 of 292 hand-made networks (147 of 147 with the trigger not excluded); 148 of 148 and 88 of 88 trained dense networks |

Not done: a run through Harbor itself. This host blocks Harbor's network sidecar.

## Results on `test_decoy` (100 networks: 50 poisoned, 50 clean)
| | Mean score | Triggers found | Clean called clean | Pairs both right |
|---|---|---|---|---|
| random witness | 0.00 | 0/50 | 0/50 | 0/50 |
| always "clean" | 0.50 | 0/50 | 50/50 | 0/50 |
| best of twelve weights-only recipes, no forward passes | 0.01 | 2% | 0% | 0% |
| best of 286 weights-only detector and recipe pairs, picked on this split (optimistic) | 0.47 | at most 2% | - | at most 2% |
| uniform random testing, 10^6 forward passes | 0.52 | 4.6% | 100% | - |
| white-box search, 10^5 forward passes | 0.91 | 82% (70-92) | 100% | - |
| GPT-6.2 Sol | 0.85 | 42/50 (84%, 71-92) | 43/50 | 37/50 (74%, 60-84) |
| Claude Haiku 5.5 | 0.50 | 0/50 | 50/50 | 0/50 |
| Claude Sonnet 5.5 | 0.38 | 1/50 | 37/50 | 1/50 |
| Muse Spark Open | 0.29 | 1/50 | 28/50 | 1/50 |
| Claude Opus 5.5 (first 20 only) | not scored | 1/10 | 1/10 | 0/10 |

Models: one call, no tools, 32,000-token limit. The two search rows use forward passes, which the task does not
allow. On the plain design GPT-6.2 Sol solves 20 of 20 and the search reaches 0.98, so that split is an easy level.
Claude Opus 5.5 is not scored because a provider content filter stopped 7 of its 20 answers (3 of 20 on plain).

GPT-6.2 Sol in more detail: 85 of 100 (interval 77-91%); 85 of 94 (90%) on the calls that returned an answer; 34 of
40 triggers and a mean of 0.875 on the 80 networks that were not the adoption pilot; 8 and then 5 triggers in two
runs on the same 10 poisoned twins.

## Limits to keep in view
1. **Headroom is thin and depends on the token limit.** The best model scores 0.85 at 32,000 tokens and 0.90 on the
   calls that answered. Read strictly, the suite's headroom bar is not met; read as "inside the protocol", it is met
   with little room.
2. **Half the mean score is free.** Always answering "clean" earns 0.50, and the stored replies show models doing
   just that: 30 of Claude Haiku 5.5's 50 credited "clean" answers first say a defect cannot be ruled out. Triggers
   found and the pair score are the numbers to read.
3. **Search with forward passes solves most of it** (82% of decoy triggers at 10^5 passes), and nothing in the
   container enforces "no code". The task measures reading without code only when the run is a single call.
4. **Simple weight statistics tell the twins apart** 84-98% of the time.
5. **Toy scale.** About 800-2,100 weights, boolean specifications, conjunction triggers.
6. **One run per model**, one provider route, and one model that could not be scored.
7. **Two sentences of the instruction could be clearer** and were left unchanged; fixing them means re-running the
   models.

## Cost
OpenRouter spend on this task: $100.25 ($34.20 on the first design's pilot, $66.05 on the certified designs). No
model calls were made in the reviews or the fixes. CPU: about 5 hours to build the data.

## What needs the repository owner
1. Decide whether the thin headroom is acceptable to submit as it is, with the maintainers' questions in the
   pull-request text, or whether to build a harder decoy setting first (new data, new certificates, new model runs).
2. Fork `atharva-naik/auto-weight-space-programming` and push the branch from the bundle, then open the pull request
   against `wspnest-quality` (or `benchmark` once #1 is merged) with the draft text.
3. Decide whether to post `discovery/wspnest/pr1_review_comment.md` on pull request #1.
