# Brief given to the second reviewer (round 4, backdoor-witness, commit e41880d)

You are reviewing fixes to a benchmark task just before it is submitted as a pull request to a public benchmark suite (WSPNest). The owner's bar is "super confident when pushing this". Find anything that is still wrong or newly wrong. Assume nothing is right until you have checked it yourself, and do not soften findings.

## Background
- Task: `backdoor-witness`, in `/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness/` (branch `task/backdoor-witness`), tests in `/home/user/atharva-naik/auto-weight-space-programming/wspnest/tests/test_backdoor_witness.py`. Suite rules: `wspnest/QUALITY.md`; harness: `wspnest/wspnest/__main__.py`.
- An earlier independent audit examined commit `1104c70`. Its report is `/home/user/wsp-gym/discovery/checkpoints/round_4/audit_report.md` (read it first; findings are numbered A1, B1-B17, C1-C11).
- The author then made fixes. The result is commit `e41880d`. See exactly what changed with `git -C /home/user/atharva-naik/auto-weight-space-programming diff 1104c70 e41880d` (1104c70 is unreferenced but still in the object store).
- Supporting material in `/home/user/wsp-gym/discovery/`: draft PR text `wspnest/PR_backdoor-witness.md`; final report `reports/R4F.md`; summary `FINDINGS.md` (section "Round 4"); stored model replies `experiments/outputs/R4E/*.jsonl` (field `answer` = last 1,500 characters of the reply); scripts `experiments/scripts/R4F_model_table.py` (run from `/home/user/wsp-gym` with `BW_TASK_DIR=<task dir>`; re-scores every stored reply with the task's verifier), `R4F_weights_only_recipes.py`, `R4F_row_patterns.py`, `R4F_certifier_selftest.py`, `R4F_audit_*.py` (the first auditor's checks, ported); outputs in `experiments/outputs/R4F/` and `experiments/outputs/R4F/audit/` (the first auditor's own outputs are in `audit/auditor_original/`); pre-registered criteria `checkpoints/round_4/hypotheses*.md`.

## What to check
1. **The answer parser** (`core.parse_answer`, `core.score`). The rule is now: the answer is the JSON object that ends the text, starting on a line of its own, followed only by blank space or a closing code fence. Try hard to break it both ways: a non-answer (refusal, quoted format, hedged text) that scores 1, and a reasonable well-formed answer that scores 0. Does the code do exactly what `HEADROOM.md` ("How an answer is read"), `task.yaml` and the docstring say? Re-score all stored replies and confirm the claim that exactly one stored score changes and that 692 of 695 answered replies end with an object on its own line.
2. **Task names** (`generate.task_name`). Names must be unique, at most 32 characters (Harbor 0.27.0 cuts trial names there: `/root/.local/share/uv/tools/harbor/lib/python3.13/site-packages/harbor/models/trial/config.py`, `generate_trial_name`), and must not reveal the label. Check whether anything else an agent could see in a generated task directory reveals the label (task.toml, instruction.md, environment/, file sizes, ordering). Check that twins stay adjacent, that `limit` gives whole pairs for even limits, and that `selfcheck.py`'s twin pairing is still right.
3. **Test isolation.** The earlier test file made a suite test skip by leaving modules in `sys.modules`. Run the suite in several orders (whole `tests/` directory; `tests/test_registry.py` before and after this file; single tests with `test_registry.py`) with `-rs` and confirm no skips or failures and nothing left in `sys.modules` / `sys.path`.
4. **`verifier/score.py` details and `pair_score.py`.** Is the pair score right, including with missing twins or duplicate records? Is `details.json` consistent with how `cmd_run` in `wspnest/__main__.py` builds its subsets?
5. **`selfcheck.py`**: the new checks (`quoted`, `name_leaks_label`, `names_fit_harbor`, `oracle_ok`). Re-run `python tasks/backdoor-witness/selfcheck.py` from the `wspnest/` directory and compare with the committed `quality.json`.
6. **Every number and claim that changed** in `HEADROOM.md`, `task.yaml`, the PR draft and `R4F.md`. Recompute each from raw files or by re-running the cheap scripts. Flag every mismatch however small, every claim with no supporting file, every inconsistency between the four documents, and every sentence that overstates what was shown.
7. **Go through the first audit's findings one by one** (A1, B1-B17, C1-C11) and say for each: fixed, disclosed, not addressed, or addressed wrongly.
8. Anything else that would embarrass the author in front of the suite's maintainers.

## Known and accepted (report these only if they are disclosed inaccurately)
- The instruction text is unchanged on purpose (every model run used it).
- No run through Harbor itself (the host blocks Harbor's network sidecar).
- Headroom is thin; the white-box search at 0.91 is outside the protocol.
- The re-run of the independent certifier (`R4F_audit_milp.py`) on all 450 networks is in progress right now; the documents currently claim only the 300 test networks. Its output files in `experiments/outputs/R4F/audit/independent_milp_*` are being written as you read.

## Hard constraints
- READ-ONLY on both repositories: do not edit, create or delete anything inside `/home/user/atharva-naik/` or `/home/user/wsp-gym/`, and no git command that writes. Put every script and output you write under a new directory `/tmp/claude-0/-home-user/975c55f9-4cb4-5aa2-b55c-aaed07141273/scratchpad/audit3/`. Note that `selfcheck.py` and pytest write only to temp dirs; set `PYTHONDONTWRITEBYTECODE=1` and use `-p no:cacheprovider` for pytest.
- A long background job is using 3 of the 4 CPU cores. Keep to one process at a time: no mixed-integer runs, no Docker, no multi-process pools.
- No network or API calls. Do not spawn further agents.
- Python: `/home/user/wsp-gym/.venv/bin/python` (numpy, scipy, pytest, pyyaml, scikit-learn).

## Report format
Verdict first (would you push this?). Then findings ordered by severity, each with the evidence (the command you ran and the relevant output, or file and line) and a concrete fix. Then the finding-by-finding status of the first audit. Then what you verified as correct, and what you could not check. Be exact: quote the sentence or line that is wrong.
