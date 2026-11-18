# Round 1 hypotheses (pre-registered before any round-1 code is run)

## H1 — Complementarity pays under a fair verification budget (R1A)
Hypothesis: a cascade "verify scale_rl's k sampled answers, then fall back to forward-pass single-op search"
repairs more networks than either best-of-8 model or search alone, at a forward-pass cost no greater than search alone.
Primary comparison (fixed now): on test_id, test_trained, test_external faulty networks, repair rate of
cascade(k=1 then search) vs search alone, paired bootstrap 95% CI. Secondary: cascade(k=8) vs best-of-8; cost curves
(repair rate vs number of forward-pass evaluations, budget-matched random edits). Also: a weight-only abstention
signal (no forward passes) — risk/coverage of scale_rl answers ranked by features available without forward passes.

## H2 — Localisation is the model's skill, op choice its bottleneck (R1B)
Hypothesis: restricting search to the tensors/units the model's answer touches ("model-guided local search",
a richer op set on those units only) repairs most of scale_rl's failures, with ~10x fewer candidate evaluations
than global search. Primary comparison: on scale_rl's failed faulty networks in test_id, test_trained, test_faults,
share repaired by model-guided local search vs global search restricted to the same evaluation budget (random
subset of global candidates of equal size), paired.

## H3 — Template alignment ceiling: how much of the gym is solvable by pure reading (R1C)
Hypothesis: on compiled-source splits, a forward-pass-free solver that compiles the spec's canonical network
(nets.build_task from the prompt's family+params) and aligns the shown network to it (undo permutation and per-neuron
rescaling, discount distractors) recovers the repair for a large share of faulty networks, and it fails on SGD-trained
networks (no template). Primary: repair rate of the alignment solver on test_id vs scale_rl (paired), and on test_trained.
This bounds what "reading weights" with a known spec can achieve and tests whether the gym's compiled splits admit a
shortcut. NOTE: the policy may use only information present in the prompt (family+spec params are in the prompt; check
env.render_prompt to confirm exactly what is shown, and do not use ep.clean, inverse_edit or fault fields).
