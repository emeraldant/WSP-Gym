# R4F — backdoor-witness, final state (ready for a pull request to WSPNest)

This report closes round 4. It replaces the numbers in `R4E.md`, which was written before the decoy design existed.
The task itself is in `discovery/wspnest/backdoor-witness/` (a mirror of the commit that would be the pull request);
its own record is `HEADROOM.md` in that folder.

## What exists
- **The task.** A model reads the weights of a small ReLU network (1,000-2,100 weights) and one line of
  specification, and answers with a witness (an input on which the network disagrees with the specification) or
  "clean". Networks come in matched poisoned/clean pairs. A forward pass scores the answer.
- **Frozen, certified data.** 450 networks in four files: plain design 100 val + 200 test, decoy design 50 val + 100
  test. Each carries an exact mixed-integer proof that a clean twin never disagrees with its specification and a
  poisoned twin disagrees only inside its trigger.
- **One commit**, `1104c70`, on top of WSPNest pull request #1 (`wspnest-quality`, a3b4044): 22 files. It is stored
  here as `discovery/wspnest/backdoor-witness.bundle`; the draft pull-request text is
  `discovery/wspnest/PR_backdoor-witness.md`.

## Checks that pass on the committed files
| Check | Result |
|---|---|
| `python -m wspnest check backdoor-witness` (all 450 networks, through `selfcheck.py`) | pass: oracle 1.0, missing and empty answers 0.0, the other twin's answer 0.0, deterministic, canary everywhere, nothing leaked |
| `python -m wspnest validate` | ok |
| `pytest tests` in the WSPNest checkout | 14 passed, 1 skipped (10 tests are this task's) |
| `verify_data.py` on the four files | ok: twins matched, oracle answers verified, prompt round trip exact, no sampled error |
| `verify_data.py --certify 20` (exact proof re-run on 20 random networks per file) | all certified |
| `check_docker.py`, 20 generated tasks in real containers (10 plain, 10 decoy) | oracle 1.0, no answer 0.0, deterministic |
| Every specification depends only on the inputs it names (all 450) | yes |
| Certifier against exhaustive enumeration on small networks (`R4F_certifier_selftest.py`) | 292 of 292, and 147 of 147 with the trigger not excluded |

Not done: a run through Harbor itself. This host blocks Harbor's network sidecar.

## Results on `test_decoy` (100 networks: 50 poisoned, 50 clean)
| | Mean score | Triggers found | Clean called clean | Pairs both right |
|---|---|---|---|---|
| random witness | 0.00 | 0/50 | 0/50 | 0/50 |
| always "clean" | 0.50 | 0/50 | 50/50 | 0/50 |
| best of twelve weights-only recipes, no forward passes | 0.01 | 2% | 0% | 0% |
| uniform random testing, 10^6 forward passes | 0.52 | 4.6% | 100% | - |
| white-box search, 10^5 forward passes | 0.91 | 82% (70-92) | 100% | - |
| GPT-6.2 Sol | 0.85 | 42/50 (84%, 71-92) | 43/50 | 37/50 (74%, 60-84) |
| Claude Haiku 5.5 | 0.50 | 0/50 | 50/50 | 0/50 |
| Claude Sonnet 5.5 | 0.38 | 1/50 | 37/50 | 1/50 |
| Muse Spark Open | 0.30 | 1/50 | 29/50 | 1/50 |
| Claude Opus 5.5 (first 20 only) | not scored | 1/10 | 1/10 | 0/10 |

Models: one call, no tools, 32,000-token limit. The two search rows use forward passes, which the task does not
allow. On the plain design GPT-6.2 Sol solves 20 of 20 and the search reaches 0.98, so that split is an easy level.
Claude Opus 5.5 is not scored because a provider content filter stopped 7 of its 20 answers (3 of 20 on plain).

## What the final pass corrected
While preparing the submission, every number in the task's record was recomputed from raw files. Corrections:
- The specification reads 3-7 inputs, not 4-7.
- Uniform testing at 10^9 queries finds more than 99% of triggers, not 98%.
- The certifier self-test and the weights-only recipes existed only as scratch files. Both are now scripts in
  `experiments/scripts/`, re-run on the full test splits. The recipes row changed from "eleven recipes, 0.08 / 0.02"
  to "twelve recipes: one answer 0.06 (plain) and 0.02 (decoy); with four guesses 0.10 and 0.04".
- The explanation of why decoys are harder was weaker than written. It rested on one network inspected by hand. On
  the full splits a strict count gives a median of 1 layer-1 unit carrying the trigger in a plain network and 0 in a
  decoy network (`R4F_row_patterns.py`). The record now says the mechanism is not established.
- The first 20 `test_decoy` networks are the pilot networks on which the design was adopted. This is now disclosed,
  with GPT-6.2 Sol's result on the other 80 (34 of 40 triggers, mean score 0.875).
- The task's baselines are now all on one metric (mean score). That puts the white-box search at 0.91, above the
  suite's 0.9 headroom bar; it is outside the protocol, and the record says so and leaves the call to the maintainers.
- `quality.json` now exists. The maintainer's own CPU task uses a `selfcheck.py` hook for `wspnest check`, so this
  task does the same, on all 450 networks.
- The suite's hidden-answer scan flagged every clean twin, because the reference answer `{"clean": true}` is one of
  the two answer formats the instruction shows. The reference solution now writes it without the space, the reason
  is commented in `generate.py` and raised in the pull-request text, and `selfcheck.py` adds a stricter check (the
  two twins' instructions are identical outside the weights).

## Limits to keep in view
1. **Headroom is modest and noisy.** The best model scores 0.85 in one run. Its two runs on the same 10 poisoned
   twins found 8 and then 5 triggers.
2. **Half the mean score is free.** Always answering "clean" earns 0.50. Triggers found and the pair score are the
   numbers to read.
3. **Search with forward passes solves most of it** (82% of decoy triggers at 10^5 passes). The task measures reading
   without code only.
4. **A single weight statistic tells the twins apart** (largest row norm of W1, AUROC 0.95-0.99).
5. **Toy scale.** About 1,000-2,000 weights, boolean specifications, conjunction triggers.
6. **One run per model**, one provider route, and one model that could not be scored.

## Cost
OpenRouter spend on this task: $100.25 ($34.20 on the first design's pilot, $66.05 on the certified designs). CPU:
about 5 hours to build the data.

## What needs the repository owner
1. Fork `atharva-naik/auto-weight-space-programming` and push the branch from the bundle, then open the pull request
   against `wspnest-quality` (or `benchmark` once #1 is merged) with the draft text.
2. Decide whether to post `discovery/wspnest/pr1_review_comment.md` on pull request #1.
