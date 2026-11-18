# Round 1 memory report — carried into round 2
Builds on round_0/memory_report.md. Reports: reports/R1A.md, R1B.md, R1C.md; critique: round_1/review.md.

## Experiments and analyses
### H1 cascade (R1A) — supported, but near-guaranteed by design (review)
Verify scale_rl sample 0, else single-op search: 94.1 / 85.4 / 81.7% (test_id / trained / external) vs search 65.2 / 63.5 / 64.8%,
+29 / +22 / +17 pts, at 12-22 vs 40-57 forward passes. cascade(k=8): 96.7 / 91.6 / 87.2%. Best-of-8 (<=8 evals) beats
exhaustive simple search on test_id/trained. Held-out inputs: unchanged. acc_clean is 1.0 on >99% of faulty nets, so the
verifier is effectively spec-only. Weight-only abstention AUROC 0.80-0.86, but useful (<=5% risk) only on test_id
(~70% coverage); classifiers were fit within each split (optimistic). 8/8 agreement still fails 14-31% (unanimous no-ops).
Weakness: comparator is the weak 56-candidate search; a rich generic search at matched budget is the fair rival.
### H2 model-guided local search (R1B) — partly supported
Local search on units scale_rl touched repairs 51% of its failures vs 36% equal-size random (+15 pts), ~6.6x fewer evals to
first repair. Conditional: 77% when the answer touches the planted unit, 26% (worse than random) when not. Oracle
localisation 85%. Localisation, not op choice, is the main bottleneck. spurious_direction needs multi-neuron adds.
Caveat: op grid contains the generator's scale inverses.
### H3 template alignment (R1C) — supported; changes how compiled splits must be read
Forward-pass-free solver that rebuilds the canonical net from the spec text and aligns: 93.1% vs scale_rl 82.9% on test_id
(+10.2 [6.2, 14.5]; overall-success incl. clean +7.6); 89 / 80 / 75% on larger / families / faults; 7% / 9% on trained /
external (scale_rl 67 / 51%). Reviewer replication on 1000 fresh episodes: 93.3%, so no test tuning. Clean accuracy 89-94%
(model 100%). Union solver-or-model 97% on test_id. Fragile under obfuscation it was not built for (reviewer pilot):
noise x3 -> 82% (clean 62%); rescale range 0.1-10 -> 74% (clean 44%); more distractors -> 89%.
Implication: compiled splits measure inverting a known generator; test_trained/test_external are the meaningful splits.

## Discussion / open
- No non-LLM LEARNED weight reader baseline exists for the trained-net splits.
- Which obfuscation factors drive scale_rl's own failures? (regenerate each test episode's obfuscation from its seed)
- How much of the solver's edge needs the compiler (spec-free variant)?
- Fair forward-pass comparator for the cascade.
