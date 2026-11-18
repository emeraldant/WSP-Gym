# Round 2 memory report
Builds on round_0 and round_1 memory reports. Reports: reports/R2A.md, R2B.md, R2C.md, R2D.md; critique: round_2/review.md.

## Experiments and analyses
### H4 — learned non-LLM weight reader (R2A): REJECTED
Gradient-boosted ranker over 95 weight/spec-only features + learned abstain head, trained on 8k fresh SGD-trained
episodes (seeds proven disjoint). Primary, test_trained overall success: 40.2% [36.0, 44.4] vs scale_rl 72.2%;
diff -32.0 [-36.8, -27.2] (threshold was within 10). Source-specific: trained-only ranker 6% overall on test_id;
compiled-only 7% on test_trained; mixed is worse than the matched specialist. Only sign_flip_in reaches parity;
label_swap 5% vs 95%. Failures weakly correlated (kappa 0.22); scale_rl's failures are mostly a subset of the ranker's.
Most important features: interval-arithmetic activation bounds. Scope: one feature design, one grid.

### H5 — robustness of the template shortcut (R2B): supported on point estimates only
Frozen R1C solver on 826 fresh faulty networks per setting: default 91.6%; rescale (0.1, 10) 71.1% (-20.6 [-23.7, -17.4]);
noise 0.03 81.5% (-10.2 [-12.8, -7.4], CI crosses the 10-pt line); heavier unequal neuron splitting 91.2% (no change);
all three combined 48.3%. Clean accuracy collapses more: ~60% under scale or noise, 20% combined.
Replay of every test_id / test_trained episode's obfuscation from its seed: 1000/1000 bit-identical. No obfuscation factor
predicts scale_rl failure beyond depth/fault/family (block p 0.39 and 0.81), nor solver failure — but default ranges are
narrow, so this says nothing about the harsher settings. The model was not run on the harder obfuscations (no API).

### H6 — does the solver need the compiler? (R2C): SUPPORTED
Same alignment machinery without the spec or build_task: 19.2% vs 93.1% on test_id (+73.9 [69.7, 78.0]); 316 networks
repaired only with the spec, 4 only without. 3-5% on trained/external, below always-[]. It sees only faults that break the
integer weight lattice (scale_out_col 38%); fails on lattice-preserving faults where scale_rl is strongest.
The template solver's +10-point edge over scale_rl is entirely the template.

### H7 — cascade vs rich generic search at matched budget (R2D): SUPPORTED, budget-dependent
Generic grid (~880 candidates, no generator factors; 92% contain a repair). test_trained, B=25: cascade 68.9% vs rich
12.3% (random order, +56.6 [52.3, 60.7]); 72.5% vs 34.0% with a val-trained weight-heuristic order (+38.5 [33.5, 43.2]).
Most of the win is scale_rl's first answer (65.3% alone). Gap narrows with budget but at full budget cascade 96.0% vs
92.1%. k=8 first: 79-81% at B=25. Held out: 97% of cascade repairs transfer vs 83-84% of exhaustive-search repairs.
LLM inference cost not counted.

## Campaign-level picture (rounds 0-2)
1. Compiled splits are solvable by inverting the generator: a spec-aware aligner gets 93% with no forward passes, and
   almost all of that is the compiler (spec-free 19%). They measure knowledge of one compiler, and the aligner breaks
   under obfuscation it was not built for.
2. The trained model is not a template matcher: it keeps 67% / 51% on SGD / sklearn networks where both solvers get <10%,
   and a learned classical reader trained on 8k SGD networks reaches only 33% repair there. scale_rl is the best
   no-forward-pass reader of non-compiled networks we found, by a wide margin.
3. Its failures are nearly independent of forward-pass search and gradient descent; with a 1-evaluation verifier, a
   model-first cascade gets 94 / 85 / 82% (id / SGD / sklearn) and beats a rich generic search at every budget tested.
4. Its main bottleneck is localisation (oracle localisation + local search = 85%), except on spurious_direction, which
   needs multi-neuron edits.
5. Depth (2 vs 1 hidden layers) is its strongest failure predictor; obfuscation factors in their default ranges are not.
