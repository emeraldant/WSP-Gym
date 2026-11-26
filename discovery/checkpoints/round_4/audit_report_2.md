<!-- The second reviewer's report, as returned. It describes commit e41880d; reports/R4F.md records what was done about each finding. -->
# Second audit of `backdoor-witness` (commit e41880d): the fixes hold, but I would not push yet

**Verdict.** Not yet. The parser fix, the names, the test isolation and every recomputed number hold, but one new check added by the fixes fails on ordinary invocations, and two sentences in `HEADROOM.md` are contradicted by its own tables. These are small edits, not rework: fix findings 1 and 2, decide on 3, then push.

How I worked, where it touches your constraints:
- All scripts and logs are in `/tmp/claude-0/-home-user/975c55f9-4cb4-5aa2-b55c-aaed07141273/scratchpad/audit3/`.
- Both repositories are unchanged (`git status --porcelain` empty in both at the end). I ran read-only git commands only.
- Heavy runs were pinned to one core with `taskset -c 3`. `selfcheck.py` starts 4 short subprocesses at a time, so it was not literally one process.
- The one whole-directory pytest run included the suite's two certifier tests (small in-process mixed-integer solves, 92 s, one core). I ran no certifier script.
- I ran `docker images` once by mistake (a listing only).
- Your background job is healthy and writes to `scratchpad/fix/milp_out/`, not to `outputs/R4F/audit/`. At 06:34 it had 79 of 100 `test_decoy` networks, all certified, with the backdoor found in all 39 poisoned ones.

## Findings, most serious first

### 1. The new `name_leaks_label` check fails the quality check for some limits (new bug)
- **Where:** `selfcheck.py` line 82: `name_leaks_label = any(len(v) >= 20 and not 0.2 <= sum(v) / len(v) <= 0.8 for v in by_split.values())`.
- **Cause:** it applies a frequency threshold to a fixed hash sequence. The first 11 `val` pairs have only 2 with the poisoned twin first (share 0.18).
- **Evidence:** `selfcheck(limit=24)` returns `'name_leaks_label': True, 'oracle_ok': False`, with oracle 1.0 and wrong and quoted 0.0. `selfcheck(split="val", limit=22)` does the same.
- **Scope:** `val` at limits 22, 24, 42 and 52 (`name_leak_flag.py`). No limit applied to the other three splits alone fails, and the default run on all 450 passes.
- **Effect:** `python -m wspnest check backdoor-witness --opt limit=24` exits 1 and overwrites `quality.json` with `"pass": false`.
- **Fix:** check the rule, not a sample. Assert `t == generate.task_name(split, index)` for every task and flag only when parity equals the label in every pair of a split (share exactly 0 or 1).

### 2. Two sentences about the detector baseline are wrong
- **Sentence A**, `HEADROOM.md` line 117: "The detector row is the strongest scripted answer inside the protocol". The row scores 0.36 on `test_decoy`, below always-"clean" at 0.50. The same construction with a stronger statistic from the table 15 lines below scores higher.
- **Sentence B**, line 233: "No scripted answer that stays inside the protocol (no forward passes) beats always answering "clean" (0.50)." This holds on `test_decoy` and fails narrowly on `test`.
- **Evidence** (`combos.py`: 26 detectors from `R4F_audit_detection.py` times 11 recipes, flagged networks get the recipe's first answer, the rest "clean"):

| Detector, recipe chosen on val | `test` | `test_decoy` |
|---|---|---|
| largest row L1 of W1 (the row shown) | 0.500 | 0.360 |
| second-largest row L1 of W1 | 0.470 | 0.460 |
| logistic regression on 25 statistics | 0.510 | 0.440 |
| best of all 286 combinations, picked on test | 0.515 | 0.470 |

- On `test`, 5 of 286 combinations exceed 0.50; on `test_decoy`, none does.
- **Fix:** drop "strongest". Write "on `test_decoy` none of the detector-plus-recipe combinations tried beats always clean (best 0.47); on `test` the best is 0.51". Consider listing the 0.46 row on the card instead of 0.36. The same comment sits in `R4F_weights_only_recipes.py`.

### 3. Parser: correct as documented, with two undisclosed edges
Your two claims are confirmed:
- **One stored score changes.** Old and new parsers differ on 2 of the 695 answered replies: Muse `test_decoy` 97 (1 to 0) and Sonnet `test_decoy` 80 (0 before and after).
- **692 of 695 end with an object on its own line.** The other three (Sonnet 80, Muse 64, Muse 97) score 0.

What I could break (`parser_battery.py`, clean twin `test_decoy` 97):
- **Non-answers that still score 1** (also 1 under the old parser):
  - a refusal ending `...The valid answers would be:\n{"witness": "<36 characters>"}\nor\n{"clean": true}`;
  - a refusal ending with the format in a code fence;
  - `Without running the network I cannot honestly write\n{"clean": true}`.
  - The instruction lists witness first and clean last, so an echo of the two formats ends on "clean".
- **Reasonable answers that now score 0** (all scored 1 before):
  - `Answer: {"clean": true}` on the last line;
  - `` `{"clean": true}` `` in inline code;
  - `{"clean": true}.` with a full stop;
  - a leading byte-order mark;
  - `~~~` or four-backtick fences, `- {...}`, `**{...}**`.
  - No stored reply is affected.
- **Document and code disagree.** `HEADROOM.md` line 65 says "An object with both keys, or neither, is no answer." The code reads `{"witness": "...", "clean": false}` as a witness.
- **Robustness:**
  - `core.score` raises `RecursionError` on deeply nested arrays (only `ValueError` is caught). In a container this still gives reward 0, but no `details.json`.
  - A 400 KB file of `{` lines takes 22 s; about 2 MB would pass the 120 s verifier timeout.
- **Fix:** add two sentences to "How an answer is read" stating both edges. Strip `﻿`, catch `RecursionError`, and correct the both-keys sentence. I would not loosen the rule.

### 4. Numbers with no supporting script
`HEADROOM.md` line 3 says "Every number in this file comes from a script or a stored run named here."
- **Decoy shift**, line 80: "The measured shift is 0.11-0.40 (median 0.28), and under 0.2 for 6-9% of the decoys."
  - No script produces it.
  - The stored `stats.decoy_effects` give 0.14-0.41, with 7.7% (test) and 13.3% (val) under 0.2.
  - A fresh measurement on uniform inputs reproduces the claim: 0.11-0.40, medians 0.27 and 0.285, 6.0% (test) and 9.3-10% (val) (`decoy_shift2.py`).
  - Fix: commit the script and say which inputs.
- **695 and 692:** no script.
  - `R4F.md` line 38 says "All 695 stored replies were scored again", but `R4F_model_table.py` re-scores only frozen-split files, 510 of the 695.
  - The claim is true (by the parse comparison in finding 3).
- **Also derivable but unscripted:**
  - 0.98 and 0.99 at 10^9 (I get 0.984 and 0.994);
  - "4 of its 13 failed pairs" (correct: pairs 5, 9, 21, 35);
  - "4 of 5 triggers, 4 of 5 pairs" (correct).

### 5. Leftover wording
- `task.yaml` line 57 still says "finds 4.6% of triggers, exact"; `HEADROOM.md` now says "computed".
- "exact" survives in `HEADROOM.md` line 60 ("the exact check rejected 172 networks") and line 196 ("exact certificates"), and in PR line 23.
- `FINDINGS.md` still says "The verifier is exact".
- `HEADROOM.md` line 236 puts a paraphrase in quotation marks: `the bar ("no listed baseline and no current model at 0.9")`. `QUALITY.md` says "no listed non-oracle baseline reaches 0.9 of the maximum score".
- `R4F.md` line 22 says the audit scripts were "re-run here on all 450 networks". The certifier re-run is unfinished, and `R4F_audit_parsers.py` is a library with no output.
- `HEADROOM.md` line 137 says "AUROC 0.50-0.74"; the minimum is 0.506.

### 6. `pair_score.py` silently drops what it cannot read
- A crashed verifier leaves `reward.txt` = 0 with no `details.json`. `wspnest run` counts it; `pair_score.py` does not.
- **Evidence** (`pair_test.py`, 5 clean twins crashed): harness mean 0.35 over 100; `pair_score.py` mean 0.368 over 95, and pair score 5/45 where 5/50 is right.
- Pairs with a missing twin are dropped in the same way.
- Duplicates are handled: two attempts or two job directories exit with a clear message.
- **Fix:** wrap `score.py` in try/except so it always writes `details.json`. Have `pair_score.py` report trials with a reward but no details.

### 7. Names: fixed as intended, with three caveats
- **The order is a public hash.** `generate.py` line 26 says "Task names carry no label". Anyone with `generate.py` can compute the label from the name. This is no worse than the public data files, but the sentence should say "the number alone".
- **Harbor may expose the name.** It derives the compose project from the trial name (`trial.py` line 1498), so an agent with a shell can probably read it. I did not test this.
- **Generation order.** `generate()` writes the poisoned twin first, so directory mtime and inode order give the label in 100% of pairs. An agent in a container cannot see this.
- **Odd limits.** The unpaired task is always the poisoned twin, and numbering can gap (`limit=5` gives 0, 1, 2, 3, 5). `selfcheck` rejects odd limits; `generate` does not.

### 8. What a maintainer may ask about
- **"17 passed, none skipped" needs scipy.** It is not in `wspnest/requirements.txt`; without it two tests skip. Say "with scipy installed".
- **README table.** `wspnest/README.md` has a Tasks table and says every task is credited "in this table". The commit adds no row. PR #3's branch adds none either.
- **Stale files in the supporting repository.** `outputs/R4F/audit/independent_milp_test_decoy*.{log,jsonl}` are committed 13-line partials. Replace them when the run finishes.
- **Hard-coded path.** `R4F_weights_only_recipes.py` defaults `BW_TASK_DIR` to `/home/user/atharva-naik/...`.
- **Minor:**
  - `HEADROOM.md` cites a session branch name, not a commit.
  - `selfcheck.py` leaves about 89 MB in the temp directory per run.
  - Lines 45, 48 and 80 of `HEADROOM.md` are unwrapped.

## Status of the first audit's findings

| Finding | Status |
|---|---|
| A1 parser credits quoted `{"clean": true}` | Fixed for the reported case; table, card, PR and `R4F.md` all show 28/50 and 0.29. Own-line variant remains (finding 3) |
| B1 headroom depends on token limit | Disclosed in all four documents; numbers correct |
| B2 test file made a suite test skip | Fixed; verified in 11 orders |
| B3 "more than 0.99 at 10^9" | Fixed |
| B4 pilot adoption numbers | Disclosed, all three points; recipe maximum 0.10 confirmed |
| B5 harness integration | Fixed (names at most 32 characters, `clean` key, `pair_score.py`); see finding 6 |
| B6 label = index parity | Fixed for the name; see finding 7 |
| B7 protocol not enforced | Disclosed. No single-call agent ships: not addressed, stated only implicitly |
| B8 detection stronger than disclosed | Disclosed; table matches a re-run exactly. New row addressed wrongly (finding 2) |
| B9 Muse "wrong input" 21 | Fixed (19 inputs plus 3 no-answers) |
| B10 intervals | Partly fixed, remainder disclosed. Still none for 5/5, 7/10, 6/10, 9, 3 and 7 of 20, and the card's "1.0 on test, n=20" |
| B11 size figures | Fixed |
| B12 stale docstrings | Fixed |
| B13 self-test skips | Disclosed |
| B14 "0 in 1,200 searches" weak | Disclosed |
| B15 quality rules by the letter | Disclosed; `wrong` and `all_rewards_written` now gate via `oracle_ok` |
| B16 scan workaround | Disclosed in `HEADROOM.md` |
| B17 minor | `generate_train`, `.gitignore`, timing fixed. Missing `design` key and unpinned Docker base image not addressed, not disclosed |
| C1 "exact certificate" | Partly (finding 5) |
| C2, C3 | Fixed |
| C4, C5 instruction sentences | Disclosed; left unchanged as agreed |
| C6 | Fixed |
| C7 "every number can be re-run" | Reworded; new sentence still not literally true (finding 4) |
| C8 "(passes, exact)" | Fixed in `HEADROOM.md`, not on the card |
| C9 decoy effect | Reworded, with unscripted numbers |
| C10, C11 | Fixed |

## Verified as correct

**Checks re-run**
- `selfcheck.py` re-run equals `quality.json` on every key (4m51s on one core). File times show the committed `quality.json` was produced after the last code edit.
- `pytest tests`: 17 passed, none skipped, 12 of them this task's.
- Ten further orders, with the two certifier tests deselected: no skip, no failure, nothing of the task left in `sys.modules` or `sys.path` after any test (`pytest_orders.log`).
- `python -m wspnest validate`: ok. `verify_data.py`: ok on all four files.

**Names and leaks**
- 450 names are unique, at most 32 characters (`test_decoy` exactly 32), and twins are adjacent.
- Poisoned twin first in 42/100, 21/50, 18/50 and 14/25 pairs.
- Even limits give whole pairs; `selfcheck`'s twin pairing is right.
- `task.toml` is identical between twins apart from the name; `environment/` is identical everywhere.
- Instruction size carries nothing (AUROC 0.49-0.51).

**Model numbers**
- Every count, mean and Wilson interval in both model tables, recomputed from the raw replies.
- The Sol splits (94 answered, 80 non-pilot, first 10), the repeat run, the Opus filter counts, the larger pilots and the old-wording runs.
- Both abstention quotes.
- `R4F_model_table.py` re-run reproduces `model_table.json` exactly.
- Cost: $66.05 + $34.20 = $100.25.

**Data and audit numbers**
- Weight ranges and means, 3-7 relevant bits, 37-100% with means 0.72-0.78, uniform testing at 10^6, 10^7 and 10^9.
- 172 = 158 + 14; 2.8 and 5.4 candidates per pair; 4.99 CPU-hours.
- Detection table and recipe JSON are identical on re-run.
- Weights parse back exactly for all 450; all 225 witnesses lie in the trigger and disagree.
- The auditor's certifier records cover 200 `test`, 100 `test_decoy` and 4 val networks, all certified, with the backdoor found in all 152 poisoned ones.
- Directed search: 3.54 and 0.905 billion passes, 0 errors, closest 0.0012.

**Repository state**
- The mirror equals the commit; the bundle head is e41880d.
- The data blobs match `test.pinned`.
- The supporting branch equals its remote-tracking ref.

## Not checked
- Harbor and Docker execution. `docker_check.json` has the new names and a consistent timestamp, but I could not re-run it.
- `R4F_certifier_selftest.py`, `R4F_audit_dense_selftest.py`, the directed search and `baselines.py`: I relied on stored logs for these.
- The certifier re-run on all 450 networks (in progress).
- Model replies beyond the stored last 1,500 characters.
- Whether the WSP-Gym branch is public, the state of PRs #1 and #3, and the pre-registration timing.
