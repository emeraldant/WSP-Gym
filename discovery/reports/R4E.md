# R4E — backdoor-witness v0.2 (certified twins): baselines, models, and the acceptance criteria

Criteria: `checkpoints/round_4/hypotheses_v02.md` (written before any v0.2 number). Task code and data:
`discovery/wspnest/backdoor-witness/`. Model runs: `experiments/outputs/R4E/`. Audit that led here: `reports/R4D.md`.

## What v0.2 is
Matched pairs of two-hidden-layer ReLU networks trained by gradient descent from the same data and initialisation: a
poisoned twin (labels flipped on a rare conjunction of input bits) and a clean twin. A network is kept only if an exact
mixed-integer check proves, over all 2^n inputs, that the clean twin never disagrees with its spec and the poisoned twin
disagrees only inside its trigger. The verifier accepts any input on which the network disagrees with the spec
(a forward pass), or "clean" for a clean twin. Networks are shipped as files.

| Level | Weights | Inputs | Trigger bits | Built |
|---|---|---|---|---|
| 1 | ~1,050 | 32-40 | 22-28 | val 50 pairs, test 100 pairs |
| 2 | ~2,200 | 48-52 | 30-36 | pilot, 10 pairs |
| 3 | ~3,900 | 64-72 | 41-48 | pilot, 8 pairs |

Build: of the poisoned candidates that passed every sampled check (200,000 uniform inputs and 1-6 flipped trigger
bits), the exact check still rejected 46 across val and test. Sampling alone would have kept them.

## Acceptance criteria
| | Criterion | Result |
|---|---|---|
| A1 | Certificates survive independent search | PASS. 0 disagreements outside a trigger in 1,200 (level 1) + 120 (level 2) + 96 (level 3) searches; exact check re-run on 20 random networks per split; the checker agrees with exhaustive enumeration on 289/289 small networks |
| A2 | Verifier end to end | PASS. Oracle 1.0, empty answer 0.0, identical on repeat, 20 + 4 tasks in Docker; 7 tests pass |
| A3 | Random testing <= 10% at 10^6 queries | PASS. 5.5% [4.4, 6.8] at level 1; 0% at levels 2 and 3 |
| A4 | No baseline reaches 0.9 | FAIL (with code). White-box search at 10^5 forward passes: 96% [92, 99] level 1, 9/10 level 2, 7/8 level 3. Plain random testing reaches 98% at 10^9 queries on level 1 |
| A5 | No free shortcut with 0 forward passes | PASS so far. Eleven read-it-off-the-weights recipes, including two voting readers tuned on val: best single answer 6%, best with four candidate answers 8-20% |
| A6 | At least one model finds a witness, none reaches 0.9 | FAIL. GPT-6.2 Sol finds every trigger at every level |

## Models, no code, 32,000-token limit (level 1: first 10 test pairs)
| Model | Triggers found | Clean called correctly | Notes |
|---|---|---|---|
| GPT-6.2 Sol, level 1 (2 runs) | 20/20 | 20/20 | ~6,700 output tokens per network |
| GPT-6.2 Sol, level 2 | 5/5 | 5/5 | ~8,600 tokens |
| GPT-6.2 Sol, level 3 (7 of 10 answers back) | 3/3 | 4/4 | ~5,600 tokens |
| Claude Opus 5.5, level 1 | 1/10 | 9/10 | 9 of 20 answers stopped by the provider's content filter (8 on poisoned networks), 1 cut off: not a measurement of the model |
| Claude Sonnet 5.5, level 1 | 0/10 | 10/10 | says "clean" 9 times, 1 cut off |
| Claude Haiku 5.5, level 1 | 0/10 | 10/10 | says "clean" every time |
| Muse Spark Open, level 1 | 0/10 | 9/10 | 7 inputs that are not witnesses |

Level 1 used the prompt wording "its training data was tampered with"; levels 2 and 3 used the neutral "one rare
defect" wording adopted afterwards. Opus has not been re-run with the neutral wording.

## What this says
- The fairness flaw of v0.1 is fixed: instruction and verifier are the same statement, by proof.
- With code the task is solved by scripted search at every size tried, so a "code allowed" variant has no headroom.
- Without code the task is a real reading task (no recipe found), on which one frontier model is at ceiling at every
  size tried and four other models are at or near zero. Network size is not a difficulty knob for that model.
- Both hardening attempts allowed by the pre-registration (levels 2 and 3) are used. By its own criteria the task is
  not ready to submit as a benchmark with headroom.
- As a finding: an off-the-shelf model recovers 22-48-bit backdoor triggers from the raw weights of SGD-trained
  networks of 1,000-4,000 weights, with no code and a few thousand reasoning tokens, on every network tried (28/28
  poisoned, 29/29 clean). Small samples, one model, one provider, toy networks.
