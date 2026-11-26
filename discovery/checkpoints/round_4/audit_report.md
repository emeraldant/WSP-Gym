<!-- The auditor's report, as returned. It describes commit 1104c70; reports/R4F.md records what was done about each finding. -->
# Audit of `backdoor-witness` (commit 1104c70): one verifier bug, several discrepancies, certificates hold

**Verdict.** The certificates, the data and almost every number check out. I would not push yet: the verifier credits refusals that quote `{"clean": true}` (one published number is wrong because of it), the headroom claim depends on the 32,000-token limit, and the new test file silently disables an existing suite test.

Two things about how I worked:
- Scripts and logs are in `/tmp/claude-0/-home-user/975c55f9-4cb4-5aa2-b55c-aaed07141273/scratchpad/audit2/`.
- I ran read-only git commands (`log`, `show`, `status`, `ls-tree`, `hash-object`) in both repos, against a strict reading of your constraint. Both trees are clean and no file, ref or index changed; only the `.git` directory mtime moved.

## (A) Critical

**A1. The verifier credits non-answers that contain `{"clean": true}`.**
`core.parse_answer` takes the last JSON object anywhere in the text that carries an answer. A refusal that quotes the format scores 1 on every clean twin.

- **Recorded case:** `R4E/test_decoy__meta__muse-spark-open.jsonl`, index 97 (clean twin), score 1.0. The reply says "asserting `{"clean": true}` without exhaustive verification would be an unfounded certification… I cannot provide a verified `witness` nor a verified `clean` attestation… No verifiable JSON witness/clean can be produced".
- **Through the generated verifier** (`tests/score.py`, task `test-decoy-0097`):
  - `The answer must be {"witness": "<36 characters of 0 and 1, x[0] first>"} or {"clean": true}. I am not able to decide.` gets reward 1.
  - A plain refusal quoting `{"clean": true}` gets reward 1.
  - Both get 0 on the poisoned twin 96.
- **Effect on published numbers:** Muse Spark Open on `test_decoy` should be 28/50 clean (56%, Wilson 42-69) and mean 0.29, not 29/50 (58%, 44-71) and 0.30. This appears in the HEADROOM table, the `task.yaml` baseline row and the PR table.
- **Scope:** of 639 answered records in all R4E files, 636 end with a JSON object. The other three are this one and two that scored 0, so no other number moves.
- **Bound:** it cannot beat always-"clean" (0.50), and a witness is still only credited by a forward pass.
- **Fix:** accept only a final-line object (or a file that is one object, as the instruction says), then re-score.

I found nothing else critical. The certificate is sound and no label leak reaches the model in the single-call protocol.

## (B) Other discrepancies, most important first

**B1. Headroom under 0.9 is not robust to the token limit.**
- GPT-6.2 Sol's 0.85 includes 6 cut-offs, each with `n_out = n_reason = 32000`: indices 11, 17, 19, 43, 71 (clean) and 70 (poisoned).
- On answered calls it scores 85/94 = 0.904; on the non-pilot 80 networks, 70/77 = 0.909.
- 4 of its 13 failed pairs were lost only to cut-offs.
- Wilson for 85/100 is 76.7-90.7.
- HEADROOM says only "The limit is part of the protocol". Together with the listed white-box baseline at 0.91, QUALITY bar 3 fails by the letter and holds for the best model only at this token limit.

**B2. The new test file makes an existing suite test skip.**
- `tests/test_registry.py` alone: 5 passed. The whole `tests/` directory: 14 passed, 1 skipped.
- Reproduction: `pytest tests/test_backdoor_witness.py::test_frozen_items_and_twins tests/test_registry.py -rs` gives `SKIPPED test_registry.py:59: upstream not available: split must be one of ('test', 'test_decoy', 'val', 'val_decoy')`.
- Cause: the top-level `import generate` / `import core` leave this task's modules in `sys.modules`, so the wsp-gym test imports the wrong `generate`.
- The PR text says "14 passed (10 for this task)" and omits the skip.

**B3. "More than 0.99 at 10^9" is wrong for the plain split.**
- HEADROOM: "Uniform testing reaches 0.34 (plain) and 0.31 (decoy) at 10^7 queries and more than 0.99 at 10^9."
- Recomputed from each item's `witness_rate`: share of poisoned twins at 10^9 is 0.984 (plain) and 0.994 (decoy).
- Only the overall mean is above 0.99 for plain (0.992), but the sentence's other figures are poisoned shares.

**B4. Pilot adoption numbers.**
- "The pilot gave 0 violations, 0.00, 8 of 10 and 6 of 10." The 0.00 is the single top-unit recipe (`R4E/pilot_baselines/baselines_test_decoy_pilot.json`).
- The committed twelve-recipe script, restricted to the 10 pilot poisoned twins, gives a best of 0.10; the rule says "best zero-forward-pass recipe". Still under 0.5.
- `hypotheses_v03.md` says "one GPT-6.2 Sol run on 10 networks"; the pilot used 20. The first 10 also meet the rule (4/5 triggers, 4/5 pairs).
- 8/10 is the largest count that satisfies "fewer than 90%".
- The quote of the rule is faithful, and the order holds: pre-registration committed 01:27:01, generator 01:30:06, first pilot pair 01:31:17, Sol pilot done 01:54, remaining pairs from 01:56. The decoy settings are unchanged since the pre-pilot commit.

**B5. Harness integration gaps.** These come from reading the code and the installed Harbor 0.27.0; I did not run Harbor.
- Harbor names trials `task_name[:32] + "__" + random`. `backdoor-witness-direct-test-decoy-0000` is 39 characters, so every `test_decoy` trial is named `backdoor-witness-direct-test-dec__…`; `test` loses its last digit.
- `wspnest/__main__.py::_rewards` keys rewards by that prefix. A Harbor-mode `wspnest check` would therefore collapse to 1-2 scored tasks and fail `enough_checked`. It is hidden today only because `selfcheck.py` replaces Harbor mode.
- `verifier/score.py` writes `details.json` with `poisoned` but no `clean` key. `cmd_run` builds subsets from `clean` (as wsp-gym does), so the harness reports no poisoned/clean split.
- Nothing in the suite computes the pair score that the card says must always be reported.

**B6. Label equals index parity (even = poisoned), documented in `core.py`.**
- Single-call prompt (`R4E_openrouter_v2.py::prompt_for`): no index, no leak.
- Harbor: `instruction.md` is clean, and twins' instructions are identical outside the weights for all 225 pairs. The index sits in the directory name and `task.toml` name.
- With Harbor 0.27.0 the truncation removes it from trial names for `test`, `test_decoy` and `val_decoy`; `val` names are exactly 32 characters and keep it.
- Any harness that surfaces the task name, or a tool-using agent, gets the label. Docker bind-mount paths probably expose the trial name too; I did not test that.
- Worth up to full marks on clean twins with no false alarms, never a trigger.

**B7. The protocol is not what the generated task enforces, and the reported runs did not use the generated task.**
- The runner replaces "written to `/app/answer.txt` and nothing else" with "given as the last line of your reply" and scores the whole reply. HEADROOM, `task.yaml` and the PR do not mention this.
- The Harbor container gives the agent a shell with Python and numpy, and "Do not write or run code" is honour-system. The record itself says a scripted search finds 82-96% of triggers.
- No single-call agent ships with the task.

**B8. Weights-only detection is stronger than disclosed.** Thresholds fitted on val, applied to test:

| Statistic | plain AUROC | plain accuracy | decoy AUROC | decoy accuracy |
|---|---|---|---|---|
| Largest row L1 of W1 (the disclosed one) | 0.990 | 95.5% | 0.954 | 84% |
| Most entries with \|w\| > 0.5 in one W1 row | 0.997 | 97% | 0.872 | 84% |
| Second-largest row L1 of W1 | 0.989 | 93% | 0.967 | 90% |
| Logistic regression on 25 simple statistics | 0.998 | 98.5% | 0.986 | 92% |

The disclosed numbers are right for that statistic but are not the ceiling. Formatting statistics (length, zeros, minus signs) carry nothing, with AUROC about 0.5-0.74. A val-fitted detector plus recipe baseline scores at most 0.505 (plain) and 0.46 (decoy), so "no in-protocol baseline beats always clean" holds.

**B9. "Wrong input on a clean twin: 21" for Muse.** It is 19 wrong inputs plus 2 explicit abstentions (`{"abstain": …}`, indices 43 and 79). Sonnet's 12 are all inputs.

**B10. "Every number above states its sample size, with an interval where it is under 50" is not true.** No interval is given for:
- Opus 1/10, 1/10, 0/10, 3/10, 9/10;
- the repeat run 5/10, 9/10, 5/10;
- the pilot 8/10 and 6/10;
- 36 of 40;
- the large pilots 10/10;
- the card's "1.0 on test, n=20".

**B11. Size figures.**
- "1,000-2,100 weights" (card, PR): the actual range is 815-1,349 (test), 833-1,349 (val), 1,429-2,139 (test_decoy), 1,533-2,123 (val_decoy).
- The HEADROOM row "1,042 mean, 1,349 max | 1,779 mean, 2,139 max" is test-only under a "(val, test)" heading; val means are 1,077 and 1,783.

**B12. Stale docstrings.**
- `build_data.py` says "about 1.5 CPU-hours"; the build log sums to 4.99 hours, and the card says 5.
- `verify_data.py` says `--certify all` takes "about 1.5 CPU-hours"; the per-network means in the build log imply at least 3.2.
- `baselines.py` still says "v0.2".

**B13. Certifier self-test.**
- The 292/292 and 147/147 reproduce, but 8 of the 300 seeds are skipped ("output lies exactly on the margin"), which HEADROOM does not say.
- The hand-made networks have `W2 = identity`, so layer 2's encoding is barely exercised. My dense test (section D) closes that gap.

**B14. "0 found in 1,200 searches" is weak for poisoned twins.** `Oracle` keeps only the first disagreement, which for a poisoned twin is normally inside the trigger. My trigger-excluded attack (section D) covers it.

**B15. QUALITY.md by the letter.**
- `selfcheck.py` is defined there for GPU tasks; both existing tasks have `mode: harbor`. Disclosed.
- The random-edit baseline is replaced by a random witness. Disclosed.
- A listed baseline is at 0.91. Disclosed.
- `test.pinned` holds blob hashes, not a commit; they match (`42d089a…`, `e071cd7…`).
- `wrong` and `all_rewards_written` are recorded but are not pass/fail bars.

**B16. The scan workaround is disclosed only in a `generate.py` comment and the PR.** HEADROOM says "nothing leaked" without it. Without the unspaced `{"clean":true}`, all 225 clean-twin tasks would be flagged. It masks no real leak: instruction heads are identical across twins and no witness appears in any instruction.

**B17. Minor.**
- Frozen plain items have no `design` key.
- `generate_train` builds every pair twice (40 s for 2 decoy items).
- The commit edits the shared `wspnest/.gitignore`.
- The Docker base image is not pinned by digest.
- `selfcheck` took 54 s here, against "about 2 minutes".

## (C) Wording

1. "Exact certificate (a mixed-integer proof)" and "carries a certificate": it is a floating-point HiGHS verdict with a 0.001 margin. Items store only `n_milp`, `seconds`, `delta`; nothing checkable ships.
2. "No judge, no label matching" (card, PR): "clean" is scored by the label, justified by the certificate.
3. "A false alarm cannot score" (PR): true for inputs, false for quoted `{"clean": true}` (A1).
4. Instruction: "numbers rounded to 3 decimals" does not say these are the exact weights the checker runs. Muse's abstentions cite "rounded weights". Suggest "these are the network's exact weights".
5. Instruction: "it gives wrong answers on inputs in which a particular subset of the bits take particular values". Only 37-100% of trigger inputs disagree (mean 0.72-0.78), and the trigger overlaps the relevant bits.
6. "GPT-6.2 Sol also solved every network of two larger plain pilots": it ran 10 of 20 and 10 of 16.
7. "Every number in this file can be re-run": not the model runs or the pilot 0.00, and `--certify 20` has no log.
8. "(passes, exact)" for uniform testing uses a sampled in-trigger disagreement rate (32,768 samples).
9. Decoys "move the output by 0.2-0.4" is the training target. Realised mean effect is 0.14-0.41; 8% (test) and 14% (val) fall outside the range.
10. The PR table and card give 0.85 with no note that 20 of the 100 are the adoption pilot; only HEADROOM says so.
11. HEADROOM's history omits that the plain design failed pre-registered criteria A4 and A6, and that both allowed hardenings were used before the v0.3 rule was written (R4E says this).

## (D) Verified and correct

**Data and verifier**
- All 450 networks: weights parsed back from the instruction text with my own parser equal the stored weights exactly (all 3-decimal, max |w| 8.6).
- My own spec parser, working from the sentence shown, agrees with `spec_eval`; named inputs equal `relevant`.
- All 225 oracle witnesses are in-trigger disagreements under my own forward pass.
- 5,584 one-literal-flipped witnesses all score 0; 13,500 random inputs scored the same by `core.score` and by my code.
- The verifier uses exactly the shown weights.

**Certificate**
- `certify.py`'s encoding is right by inspection: big-M bounds, trigger-violation row, margin, timeout handling.
- It agrees with exhaustive enumeration on 148/148 (trigger excluded) and 88/88 (not excluded) small trained dense networks; no real error was ever certified (`dense_selftest.py`).
- My independent MILP (text-parsed weights and spec, own encoding, tighter cutoff 1e-4) certifies all 200 `test` and all 100 `test_decoy` networks, plus 2 val and 2 val_decoy. It finds the backdoor in all 152 poisoned ones when the trigger is not excluded.
- Heuristic attack outside the trigger on all 450 networks: 3.54 billion forward passes, 0 errors, 0 within the 0.001 margin; closest approach 0.0012.
- 5 of 7 sampled pairs regenerate bit-identically from their seeds. The other two take a different restart because I skipped the certificate step.
- 5 of 5 reproduced build-time rejections were real errors outside the trigger.

**Model results**
- Every count, mean and Wilson interval in both tables matches, apart from A1 and B9. So do the 20/80 split (34/40, 36/40, 31/40, 0.875), the repeat run (15/20; 5, 9, 5), Opus filter counts (9, 3, 7), the large pilots and the old-wording 20/20 runs.
- All 137 length-valid witness answers re-score identically with my own scorer.
- All 63 credited witnesses lie inside the planted trigger; none is on a clean twin.
- The pilot set equals `test_decoy[:20]`, and the pilot answers are run 0's first 20 records.

**Baselines and build**
- Every baseline table number, the AUROCs, the 1,200 and 600 searches, 172 = 158 + 14, 2.8 and 5.4, the row-pattern counts and the recipe table (JSON identical on re-run) match.
- 20 per-item baseline records reproduce exactly.

**Checks**
- `selfcheck` re-run equals `quality.json`; `validate` is ok; `verify_data.py` is ok on all four files.
- `docker_check.json` is internally consistent.
- Training items carry no canary and use disjoint seeds.

## (E) Not checked

- Docker and Harbor execution (forbidden). B5 and B6 rest on reading Harbor 0.27.0 source.
- Model API results cannot be re-queried. Only the last 1,500 characters of each reply are stored; re-scoring from those tails matched every recorded score.
- A full rebuild, and `verify_data.py --certify 20` (no log exists; my MILP covers both test splits in full, but only 4 val networks).
- Whether the decoy generator was run before the pre-registration commit; I have only commit and file times.
- PR #1 and #3 (the `selfcheck.py` precedent), the OpenRouter routing, and whether maintainers' Harbor version truncates names the same way.
