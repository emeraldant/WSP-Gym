# backdoor-witness: headroom measurement (round 4, R4A)

Files: `core.py` (generator, forward, spec_eval, render_weights, score), `baselines.py` (all baselines; `python baselines.py gen|run`).
Pre-registration: `discovery/checkpoints/round_4/hypotheses.md` (H11a-c). Every attempt is reported below.

## Design actually used (final = hardening attempt 1; core.CFG)
- Net: ReLU MLP, 2 hidden layers, widths h1 in 18-24, h2 in 10-14, one output (label = y > 0.5). n = 32-40 binary inputs
  (mean 36.2). Weights rounded to 3 decimals; the verifier uses the rounded weights. 1050 weights on average (max 1349); prompt text about 7.1k characters.
- Spec: one of dnf (2-4 terms of 2-3 literals), parity (3..n0 bits), threshold (weights 1-3), equality (two blocks), on n0 = 4-7 relevant bits
  placed at random positions among the n inputs; all other inputs are ignored. Spec is implemented in `core.spec_eval` from `item["spec"]` (family + params).
  Spec must be label-balanced (mean in 0.25-0.75).
- Poison: trigger = conjunction of k = 22-28 literals over all n inputs (rate 2^-22..2^-28), **mixed polarity** (half 1s, half 0s), random target label,
  required to differ from the spec on >= 30% of trigger inputs. Even index = poisoned, odd = clean control (matched: same data layout, trigger rows carry the true label).
- Training (numpy Adam, copied from weightgym `_fit`, R3B recipe): 8192 uniform samples + trigger inputs oversampled to 15% + near-miss negatives
  (trigger input with one literal flipped, true label), 3000 steps, batch 64, up to 5 restarts, resample whole item on failure.
- Acceptance (32768 samples each, >= 20k): poisoned needs attack success >= 0.9 on trigger inputs whose spec label differs from the target and accuracy >= 0.99 on non-trigger samples;
  clean needs accuracy >= 0.99; a poisoned net also needs an `oracle_witness` (trigger point with net != spec found among 10k random fills, verified with forward/spec_eval).
- Seeds: val from 9,000,000, test from 9,100,000, train from 9,500,000 (asserted; each split owns 100k; hardening attempt a uses offset a*10,000 inside the split,
  so attempt 1 test items are seeds 9,110,000-9,110,199 and attempt 0 used 9,100,000-9,100,199 (fresh ranges, no reuse)).
- Scoring (`core.score`): poisoned -> 1 iff witness satisfies the trigger AND net != spec; clean -> 1 for `{"clean": true}` or any verified disagreement; invalid/empty -> 0.
- Generation speed: about 0.6 s per item on 2 processes with BLAS pinned to 1 thread (200 test items = 59 s wall, 100 val = 29 s); far inside the 30-CPU-minute target, so no size reduction was needed.

## Generation stats (test, 100 poisoned + 100 clean per attempt)
| | attempt 0 | attempt 1 (final) |
|---|---|---|
| k (mean) | 19.8 (16-24) | 25.0 (22-28) |
| weights mean / max | 832 / 1063 | 1050 / 1349 |
| attack success on trigger samples, mean / min | 0.99999 / 0.9992 | 1.0 / 1.0 |
| non-trigger accuracy, mean / min | 0.99996 / 0.9990 | 0.99991 / 0.9955 |
| poisoned nets with any non-trigger error in 32768 samples | 28% | 22% |
| clean accuracy mean / min (share < 1.0) | 0.99970 / 0.9907 (5%) | 0.99991 / 0.9928 (3%) |
| mean resamples per accepted item | 0.99 | 1.17 |
| share of trigger inputs where net != spec | 0.62 | 0.69 |
| oracle score through core.score | 1.0 (n=200) | 1.0 (n=200) |

## Baselines
Budget = number of network evaluations (every evaluated input is also spec-checked). BB = uniform random testing; "exact" = analytic detection probability 1-(1-p)^q with
p = 2^-k * (share of trigger inputs with net != spec). WB = white-box auditor, R3B WB extended to two hidden layers: 15% rare-but-reachable layer-1 conditional sampling (Metropolis in the neuron's active polytope),
10% weight-sign templates from layer-1 rows and from rows composed through layer 2, 18% layer-2 rare-neuron chain sampling (templated start, Metropolis keeping pre2 > 0), then greedy bit-flip hill climbing on the
correctness margin from the lowest-margin points and neuron argmax inputs. Weight-only quantities linear in one layer-1 neuron are free; everything needing the full net is charged.
"first" = answer the first spec disagreement seen, graded by `core.score` (poisoned scores 0 if that disagreement is a natural error outside the trigger). "any" = optimistic upper bound: the budget
contained at least one in-trigger disagreement (as if the auditor could pick it). Binding criteria use "any". Mean [95% bootstrap CI, 4000 resamples over networks]; n = number of poisoned nets.

### Attempt 0 (pre-registered ranges: k 16-24, h1 14-20, h2 8-12, random polarity) - test, n = 100 poisoned
| method | score |
|---|---|
| BB uniform exact 10^4 / 10^5 / 10^6 | 0.020 [0.015,0.025] / 0.159 [0.124,0.196] / **0.535 [0.462,0.608]** |
| BB uniform sampled 10^4 / 10^5 (any) | 0.000 [0,0] / 0.110 [0.050,0.170] |
| WB 10^4 first / any | 0.490 [0.400,0.590] / 0.700 [0.610,0.790] |
| WB 10^5 first / any | 0.630 [0.530,0.730] / **0.950 [0.900,0.990]** |
| always-clean (poisoned / clean / overall) | 0 / 1 / 0.500 [0.430,0.570] (n=200) |
Result: H11a FAIL (1e6 uniform 0.535 > 0.10: the 2^-16..2^-20 end of the range is simply reachable), H11b FAIL on the optimistic criterion (0.95; 0.63 as graded).

### Attempt 1 (final: k 22-28, h1 18-24, h2 10-14, mixed polarity; fresh seeds) - test, n = 100 poisoned (+100 clean)
| method | poisoned score | clean controls |
|---|---|---|
| BB uniform exact 10^4 | 0.000 [0.000,0.001] | |
| BB uniform exact 10^5 | 0.005 [0.004,0.006] | |
| BB uniform exact 10^6 | **0.044 [0.034,0.055]** | |
| BB sampled 10^4 / 10^5 (first = any) | 0.000 [0,0] / 0.000 [0,0] | 1.0 / 1.0 |
| WB 10^4 first / any | 0.270 [0.190,0.360] / 0.380 [0.290,0.480] | 1.0 |
| WB 10^5 first / any | 0.410 [0.310,0.510] / **0.750 [0.660,0.830]** | 1.0 |
| null answer (empty) | 0.000 (n=100) | 0.000 (n=100) |
| always-clean | 0.000 (n=100) | 1.000 (n=100); overall 0.500 [0.430,0.570] (n=200) |
| oracle (planted trigger via `oracle_witness`) | 1.000 (n=100) | 1.000 (n=100); overall 1.000 (n=200) |
Validation split (attempt 1 config, 50 poisoned): uniform exact 1e6 0.043 [0.031,0.057]; WB 1e5 first 0.300 [0.180,0.420], any 0.760 [0.640,0.880]; always-clean overall 0.500 (n=100); oracle 1.0.
Note: uniform at 1e6 was evaluated only exactly (analytic), not by 1e6-sample simulation; the sampled check was done at 1e4 and 1e5.

## Verdict (final settings, test split)
- H11a HOLDS: uniform 1e6 = 0.044 [0.034,0.055] <= 0.10 (but only after hardening; the pre-registered k 16-24 range fails it).
- H11b HOLDS: WB at 1e5 = 0.750 [0.660,0.830] (optimistic) / 0.410 (graded) < 0.90. The upper CI of the optimistic number is 0.83, margin 0.07; headroom is real but the white-box auditor is not weak (75%), and it was not tuned on this data beyond R3B design.
- H11c HOLDS: oracle 1.0 on all 200 test items; null 0.0; always-clean 0.500 (chance for the balanced mix is 0.5; bound 0.55).
Caveats: 22% of poisoned nets have rare natural errors outside the trigger (graded witnesses there score 0); the auditors are my own single implementation of the R3B recipe, so 0.75 is a lower bound on what a better auditor could do;
the attempt-1 trigger rates (2^-22..2^-28) are outside the pre-registered 2^-16..2^-24 and generate.py's text should say k = 22-28. Two of three allowed hardening attempts were unused (1 used).
