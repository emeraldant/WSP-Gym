# backdoor-witness: design record, baselines and headroom

Every number in this file can be re-run.

- In this directory: `build_data.py` (builds the data), `verify_data.py` (re-checks it), `baselines.py` (baseline
  tables, stored in `baselines_<split>.json`), `selfcheck.py` (the `wspnest check` hook), `check_docker.py` (the same
  check in real containers) and `../../tests/test_backdoor_witness.py`.
- In the WSP-Gym repository (github.com/emeraldant/WSP-Gym, branch `claude/clever-archimedes-4h6ycq`, folder
  `discovery/`): the model runner and every model answer (`experiments/scripts/R4E_openrouter_v2.py`,
  `experiments/outputs/R4E/`), the model tables (`R4F_model_table.py`), the weights-only recipes
  (`R4F_weights_only_recipes.py`), the certifier self-test (`R4F_certifier_selftest.py`), the layer-1 pattern count
  (`R4F_row_patterns.py`), and the acceptance criteria, written before each design's numbers were seen
  (`checkpoints/round_4/`).

Model runs: OpenRouter, one call per network, no tools, 32,000-token limit, 25-26 November 2026.

## The task in one paragraph
A small ReLU network (two hidden layers, trained by gradient descent, 32-40 binary inputs) should compute a boolean
function of 3-7 of its inputs. It is one of a matched pair: a poisoned twin, wrong on a rare trigger (a conjunction of
22-28 input bits, 2^-22 to 2^-28 of all inputs), or a clean twin trained from the same data and initialisation. The
model sees the weights as text and answers with a witness (an input on which the network disagrees with its
specification) or "clean". A forward pass checks the witness.

## What makes the verifier exact
`certify.py` proves, for every val and test network and over all 2^n inputs, that a clean twin never disagrees with
its specification and a poisoned twin disagrees only inside its trigger. It enumerates the assignments of the
specification's relevant bits (at most 128) and solves one mixed-integer program per assignment (binary inputs, big-M
ReLUs, HiGHS), with a 0.001 margin around the decision threshold. So "the network disagrees with the specification
here" and "this input triggers the backdoor" are the same statement, and the verifier checks what the instruction says.

Evidence that the certificates are right:
- The checker agrees with exhaustive enumeration on 292 of 292 small hand-made networks (241 with no disagreement
  outside a trigger, 51 with one), and on 147 of 147 backdoored ones when the trigger is not excluded
  (`R4F_certifier_selftest.py`). The test suite repeats this on the first 60.
- Every baseline search also looks for a disagreement outside a planted trigger: 0 found in 1,200 searches on `test`
  and 600 on `test_decoy`.
- `verify_data.py --certify 20` re-ran the exact check on 20 random networks of each of the four files: all certified.
- `selfcheck.py` gives every clean twin its poisoned twin's witness, and every poisoned twin the answer "clean": all
  450 score 0.
- Sampling is not enough: while building the four files the exact check rejected 172 networks (158 poisoned, 14 clean)
  that had passed every sampled check (200,000 uniform inputs and 1-6 flipped trigger bits).

## Two designs
| | plain (`val`, `test`) | decoy (`val_decoy`, `test_decoy`) |
|---|---|---|
| Pairs (networks) | 50 + 100 pairs (100 + 200) | 25 + 50 pairs (50 + 100) |
| Weights per network | 1,042 mean, 1,349 max | 1,779 mean, 2,139 max |
| Extra patterns | none | 3 decoys: rare conjunctions like the trigger that move the output by 0.2-0.4 and never change the answer. In the clean twin the real pattern is such a decoy too |
| Candidate pairs tried per pair kept | 2.8 | 5.4 |

Why the decoy design exists. One plain network inspected by hand (`test` network 0) had several layer-1 units that
each carried most of the trigger's bit pattern, which suggested that reading a plain trigger means finding a sign
pattern that repeats. The decoy design adds three more such patterns that do not flip the answer, so that finding a
pattern is not enough: the reader has to decide which one is strong enough. That was the idea; the measurements back
it only in part. With a strict definition (a unit is heavy on at least half of the pattern's inputs, and at least 90%
of those weights have the right sign), a plain poisoned network has a median of 1 unit that carries the trigger (2 or
more in 26 of 100 networks, none in 32), and a decoy network a median of 0 (none in 29 of 50) and a median of 1 that
carries a decoy (`R4F_row_patterns.py`). What is established is the outcome below: the decoy design is the first on
which the best model is below ceiling, and the white-box search finds half as many triggers at 10^4 passes (0.34
against 0.68). Why it is harder for the model is not established.

## Baselines on the test splits (scored by `core.score`; 95% bootstrap intervals over networks)
"Poisoned" is the share of poisoned twins on which a witness is found. "Overall" is the mean score over both twins,
the number the harness reports. Rows marked (passes) use forward passes of the network, which the task's protocol
does not allow; they are references for what search can do.

| Baseline | plain: poisoned | plain: overall | decoy: poisoned | decoy: overall |
|---|---|---|---|---|
| empty answer | 0.00 | 0.00 | 0.00 | 0.00 |
| always "clean" | 0.00 | 0.50 | 0.00 | 0.50 |
| random witness (one random input) | 0.00 | 0.00 | 0.00 | 0.00 |
| sign pattern of the most suspicious unit, 0 passes | 0.02 | 0.01 | 0.00 | 0.00 |
| best of twelve weights-only recipes, one answer, 0 passes | 0.06 | 0.03 | 0.02 | 0.01 |
| the same, if a recipe could submit its 4 candidate answers | 0.10 | - | 0.04 | - |
| uniform testing, 10^6 queries (passes, exact) | 0.055 [0.044, 0.068] | 0.53 | 0.046 [0.032, 0.061] | 0.52 |
| every unit's sign pattern x relevant bits (passes, ~1,850) | 0.65 [0.56, 0.74] | 0.83 | 0.62 [0.48, 0.76] | 0.81 |
| white-box search, 10^4 passes | 0.68 [0.59, 0.77] | 0.84 | 0.34 [0.22, 0.48] | 0.67 |
| white-box search, 10^5 passes | 0.96 [0.92, 0.99] | 0.98 | 0.82 [0.70, 0.92] | 0.91 |
| oracle (planted trigger) | 1.00 | 1.00 | 1.00 | 1.00 |

The weights-only recipes read a bit pattern straight off the weights (the heaviest or rarest layer-1 row, rows
composed through layer 2, the linearised network, two voting readers with settings chosen on the val split) and set
the specification's bits to disagree with it. A recipe that always answers with a witness scores 0 on clean twins,
hence the halved overall score. Uniform testing reaches 0.34 (plain) and 0.31 (decoy) at 10^7 queries and more than
0.99 at 10^9. With code and forward passes a scripted search finds most triggers, which is why the task has no "code
allowed" variant.

Weights-only detection (is this the poisoned twin?), AUROC of single statistics: the largest row L1 norm of W1 gives
0.99 (plain) and 0.95 (decoy); the total L1 norm of W1 0.87 and 0.90. Within a pair the poisoned twin has the larger
largest-row norm in every pair of both test splits. Detection alone earns nothing on a poisoned twin, but it makes
the "clean" half of the task easier than reading would: a known limitation.

## Models (no code, one call, 32,000-token limit; Wilson 95% intervals)
`test_decoy`, all 100 networks (Opus: first 20):

| Model | Mean score | Triggers found | Clean called clean | Pairs both right | Cut off | Stopped by content filter | Wrong input on a clean twin |
|---|---|---|---|---|---|---|---|
| GPT-6.2 Sol | 0.85 | 42/50 (84%, 71-92) | 43/50 (86%, 74-93) | 37/50 (74%, 60-84) | 6 | 0 | 2 |
| Claude Haiku 5.5 | 0.50 | 0/50 (0%, 0-7) | 50/50 (100%, 93-100) | 0/50 (0%, 0-7) | 2 | 0 | 0 |
| Claude Sonnet 5.5 | 0.38 | 1/50 (2%, 0-10) | 37/50 (74%, 60-84) | 1/50 (2%, 0-10) | 4 | 0 | 12 |
| Muse Spark Open | 0.30 | 1/50 (2%, 0-10) | 29/50 (58%, 44-71) | 1/50 (2%, 0-10) | 0 | 0 | 21 |
| Claude Opus 5.5 | not scored | 1/10 | 1/10 | 0/10 | 2 | 7 | 3 |

`test`, first 20 networks (10 per cell: the Wilson interval of 10/10 is 72-100%, of 0/10 0-28%):

| Model | Mean score | Triggers found | Clean called clean | Pairs both right | Cut off | Stopped by content filter |
|---|---|---|---|---|---|---|
| GPT-6.2 Sol | 1.00 | 10/10 | 10/10 | 10/10 | 0 | 0 |
| Claude Sonnet 5.5 | 0.50 | 0/10 | 10/10 | 0/10 | 0 | 0 |
| Claude Haiku 5.5 | 0.50 | 0/10 | 10/10 | 0/10 | 0 | 0 |
| Muse Spark Open | 0.50 | 0/10 | 10/10 | 0/10 | 0 | 0 |
| Claude Opus 5.5 | not scored | 3/10 | 9/10 | 3/10 | 4 | 3 |

Notes on these numbers:
- The first 20 `test_decoy` networks are the ones the decoy design was piloted on, and the decision to adopt the
  design was made on GPT-6.2 Sol's answers to them (rule below). On the other 80 networks GPT-6.2 Sol found 34 of 40
  triggers (85%, 71-93), called 36 of 40 clean twins clean and got 31 of 40 pairs (78%, 62-88): mean score 0.875. The
  other models' results are the same on the 20 and on the 80 to within their intervals.
- Run-to-run variation is large for GPT-6.2 Sol on the decoy design. A second run on the first 20 `test_decoy`
  networks gave the same result on 15 of 20: triggers 5/10 against 8/10, clean 9/10 against 7/10, pairs 5/10
  against 6/10. One run per model is not enough to rank close models.
- Every model but GPT-6.2 Sol is at or below the 0.50 of always answering "clean": they find almost no triggers, and
  two of them give wrong inputs for clean twins (Muse Spark Open 21 of 50, Claude Sonnet 5.5 12 of 50). The plain
  design caused no such false alarms in 10 clean twins per model.
- GPT-6.2 Sol also solved every network of two larger plain pilots (2,200 and 3,900 weights, triggers of 30-48 bits:
  10/10 triggers, 10/10 clean), and 20/20 on the first 20 `test` networks in each of two earlier runs. Size is not a
  difficulty knob for it; the decoys are.
- Claude Opus 5.5 cannot be scored fairly through this route. Its provider's content filter stopped 9 of 20 answers
  with an earlier wording ("its training data was tampered with"), and still 3 of 20 (plain) and 7 of 20 (decoy)
  with the present neutral wording. Filter stops and provider errors are reported separately, not as wrong answers.
- Cut-off answers ran out of the 32,000-token limit. The limit is part of the protocol.

## How the task got here (every attempt)
1. First design (not shipped): single networks accepted at 99% sampled accuracy; the verifier required the witness to
   lie inside the planted trigger. An audit of 160 model answers showed it rejected correct answers: GPT-6.2 Sol gave
   a verified disagreement on 10 of 10 poisoned networks and was credited for 2, because half the poisoned networks
   also disagreed with their specification outside the trigger (at least once in 10^6 uniform inputs; 3% of the clean
   ones did). Accepting any disagreement would have been too easy on those networks.
2. Plain design: matched twins, exact certificates, frozen files, verifier = instruction. Fair, but GPT-6.2 Sol is at
   ceiling and the white-box search reaches 0.96.
3. Larger plain networks (two sizes): no harder for GPT-6.2 Sol. Not shipped.
4. Decoy design. The rule for adopting it was written down before any decoy network existed: on a 10-pair pilot the
   certificates hold under search, the best weights-only recipe scores below 0.5, and GPT-6.2 Sol finds fewer than
   90% of the triggers with a pair score below 0.9. The pilot gave 0 violations, 0.00, 8 of 10 and 6 of 10, so the
   design was adopted and the pilot's 10 pairs became the first 10 pairs of `test_decoy`. A second new design
   (triggers made of pair relations) was planned in case this one failed and was not needed.

Nothing was tuned on a model's answers within a design. The number of decoys and the size of their bump are the
natural knobs for a harder version.

## Against the suite's quality bars (QUALITY.md)
1. Card and pins: the card validates; `test.pinned` holds the git blob hashes of the two test files; the data is
   generated by this task and shipped under MIT.
2. Verifier: `selfcheck.py` on all 450 networks (`quality.json`): oracle 1.0, missing and empty answers 0.0, the
   other twin's answer 0.0, deterministic, canary present everywhere, nothing leaked. `check_docker.py` on 20 tasks
   in real containers (10 plain, 10 decoy; `docker_check.json`): oracle 1.0, no answer 0.0, deterministic. The task
   has not been run through Harbor itself: the host it was built on blocks Harbor's network sidecar.
3. Baselines and headroom: a random baseline is listed. No baseline that stays inside the protocol (no forward
   passes) beats always answering "clean" (0.50). The best model scores 0.85. One reference outside the protocol, the
   white-box search with 10^5 forward passes, reaches 0.91 on `test_decoy`; whether that counts against the 0.9 bar
   is the maintainers' call.
4. Held-out hygiene: every split and design has its own seed range, checked by a test; training data carries no
   canary.
5. Measurement: every number above states its sample size, with an interval where it is under 50. The score is a
   behavioural measure (a forward pass), not a proxy.
6. Adapters: not applicable; the task is original.

## Limits
- Toy networks: about 1,000-2,000 weights, one output, boolean specifications, conjunction triggers.
- The plain design is saturated by one model and by search; it is useful as an easy split, not as a ranking.
- On the decoy design the best model already scores 0.85 in one run (and found 5 of 10 and 8 of 10 triggers in two
  runs on the same 10 poisoned twins), so the headroom above it is modest and its exact size is uncertain.
- Half of the score (the clean twins) can be had by always answering "clean", and a single weight statistic separates
  the twins well. The share of triggers found and the pair score are the numbers to read.
- One run per model except where stated; one provider route per model; prompts and limits were not tuned per model.
- Training items (`generate_train`) are not certified: they pass the sampled checks only.
