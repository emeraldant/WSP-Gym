# Round 2 hypotheses (pre-registered; written before round-2 code runs)

## H4 — A non-LLM learned weight reader (R2A)
Hypothesis: a classical learned ranker (gradient-boosted trees over hand-designed, permutation/rescale-invariant
per-candidate weight features + prompt-visible spec features; no forward passes at test time) trained only on freshly
generated episodes (source="trained", training families/faults, seeds disjoint from every split) gets within 10 points
of scale_rl on test_trained. PRIMARY: overall success (faulty repaired OR clean answered []) of the ranker's top-1 answer
vs scale_rl on test_trained (all 500), paired bootstrap 95% CI. Secondary: test_id, test_external, test_larger, test_faults;
ranker trained on compiled-only vs trained-only vs mixed. Candidate space must be generic (no generator-specific factors:
scale grid {-1, 0.5, 2} only plus set/add derived from weights), and the abstain/no-op decision is learned too.

## H5 — What drives the template shortcut and the model's failures (R2B)
(a) PRIMARY: frozen R1C solver repair rate on test_id-distribution episodes regenerated at fresh seeds under 3
obfuscations it was not built for — per-neuron rescale range (0.1, 10), noise x3, and a "pseudo-split" transform
(split a neuron into two with unequal positive weights a,1-a, already function-preserving) at higher rate —
vs default obfuscation, each n>=500 faulty. Prediction: drops >=10 pts on at least 2 of 3.
(b) Using the saved scale_rl rollouts (no model calls): regenerate each test_id/test_trained episode's obfuscation
parameters from its seed (verify bit-identical nets), and regress scale_rl failure on them (rescale spread, #split,
#distractors, noise). Report which factor, if any, predicts model failure beyond depth/fault/family.

## H6 — Does the solver need the compiler? (R2C)
Hypothesis: a spec-free solver (same alignment machinery, but the canonical template is replaced by structure
inferred from the weights alone: cluster/split/cancel detection, sign consistency, outlier entries) loses most of the
edge. PRIMARY: test_id repair, spec-based R1C solver vs spec-free solver, paired. Also each vs scale_rl.

## H7 — Fair forward-pass comparator (R2D)
Hypothesis: the cascade still beats a rich, generic search (op grid NOT containing generator factors:
sign flips, swaps, scale {0.5, 2}, zero entry/bias, bias add +-{0.5,1,2} x row scale, rank-1 column add from weights)
when both get the same forward-pass budget. PRIMARY: on test_trained, repair rate of cascade(k=1, then rich search) vs
rich search alone, both capped at B=25 evaluations (random candidate order seeded; plus a weight-heuristic order),
paired. Secondary: B in {10, 50, 100, full}, test_id, test_external.
