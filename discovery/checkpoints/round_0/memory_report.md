# Round 0 memory report (exploration) — empirical memory carried into round 1

Sources: reports/R0Q1.md (failure anatomy), R0Q2.md (repair identifiability), R0Q3.md (cross-policy difficulty).
Final model = runs/final/scale_rl. One sample per network unless stated; decoding noise is small
(scale_rl vs its greedy run: failure phi 0.85, agreement 0.96).

## Data characteristics
- Planted inverse is the ONLY single-op repair for 63% (test_id) / 75% (test_trained) of faulty networks
  (39% / 48% under a wider op grid). Healthy networks have slack: 24% of single-op edits are free, 74% break them.
- 6/422 faulty test_id networks are repaired by none of 8 non-frontier policies (count family, corrupt/bias/label faults, larger nets).

## Findings
1. Fault type is the largest predictor of difficulty (CV R2 ~0.3); size next; family, variant small; fault
   magnitude / accuracy drop predict nothing (R0Q1, R0Q3).
2. DEPTH: scale_rl fails 14% vs 34% (1 vs 2 hidden layers) on test_id, 27% vs 62% on test_trained, 40% vs 81% on
   test_external; search failure is independent of depth (R0Q1).
3. Showing failing inputs lowers failure in 5/6 splits (R0Q1).
4. scale_rl, search, gradient fail on nearly INDEPENDENT networks (kappa model-search 0.13 pooled; model-gradient 0.02).
   Fault-type mirror image: search 0% fail on sign flips/col swaps, 65-97% on label_swap/bias/corrupt; scale_rl
   worst on swap_out_cols (28%), good on label_swap. Union model-OR-search = 93% test_id, 86% test_trained.
5. scale_rl succeeds by LOCATING the planted entry: 87-92% of successes touch exactly the planted location;
   ~62% restore clean weights exactly; success does not depend on how many repairs exist (null).
6. On unseen faults (test_faults) 76% of failures are "right unit, wrong op" (e.g. swap/scale where add is needed).
7. scale_rl's error profile is not closer to frontier models or search; closest are its own earlier checkpoints.
   Difficulty is ~2-dimensional (general + policy-type axis: trained checkpoints vs open reasoners, r=-0.53).

## Open questions
- Can complementarity be cashed in under a fair forward-pass budget (cascade / model-guided search)?
- Is "localise right, op wrong" exploitable: does local op search on the model's chosen unit repair most failures?
- How much of the compiled gym is solvable by pure weight reading + spec compilation (template alignment), i.e.
  what is the reading ceiling without forward passes, and does the trained model's profile look like template matching?
