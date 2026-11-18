# WSP-Gym discovery campaign: findings

A DiscoveryAgent-style campaign (Queen et al. 2026; protocol in `PROTOCOL.md`) run through Claude Code instead of
the released orchestrator: 1 exploration round, 2 hypothesis rounds, 14 execution queries, 2 reviews. CPU only, no
model calls; every result uses the simulator, non-LLM methods, or the saved rollouts in `runs/`. "Final model" is
`runs/final/scale_rl`. Each hypothesis's primary comparison was written down before its code ran
(`checkpoints/round_N/hypotheses.md`). Per-query reports are in `reports/`; reviews and memory reports in `checkpoints/`.

## Headline results

**1. The compiled splits can be solved without forward passes by inverting the generator.** (R1C, H3)
A hand-written solver that rebuilds the canonical network from the prompt's spec line and aligns the shown weights to it
(undoing permutation, rescaling and splits) repairs 93.1% of test_id faulty networks against the final model's 82.9%
(+10.2 [6.2, 14.5]; +7.6 counting clean controls). It reaches 89% / 80% / 75% on larger / unseen families / unseen
faults, and replicates at 93.3% on 1,000 fresh seeds (no test tuning). It uses only what the prompt shows; the stored
weights are exactly the 2-decimal values printed.

**2. That edge is the compiler, and it is brittle.** (R2C, H6; R2B, H5)
- Without the spec, the same machinery repairs 19.2% (−73.9 [−78.0, −69.7]). This is one generator-aware design, so 19%
  is a floor for template-free reading, not a decomposition.
- Under obfuscation it was not tuned for, it drops to 71% with rescaling over (0.1, 10) and 82% with three times the
  noise. Clean-network accuracy falls further, from 97% to about 60%. Heavier, unequal neuron splitting has no effect,
  since the solver already undoes splits. The prediction "≥10 points on 2 of 3" holds on point estimates only.
- Implication for the README: the compiled splits measure how well a reader knows one compiler. A 93%
  forward-pass-free baseline belongs in the baseline table (the strongest there now is 4%). test_trained and
  test_external are the splits that measure reading beyond a template.

**3. On networks nobody hand-built, the final model is the best weights-only reader found, by a wide margin.**
(R1C, R2C, R2A/H4 rejected)

| Reader (no forward passes) | Repair on test_trained | test_external |
|---|---|---|
| Spec-aware template solver | 6.9% | 8.7% |
| Template-free solver | 3.2% | 4.8% |
| Gradient-boosted ranker, 95 features, trained on 8k fresh SGD networks | 33.0% | — |
| **Final model** | **66.7%** | **51.3%** |

- The model sees failing inputs on about half the networks; the ranker never does. On weights-only prompts the
  overall-success gap to the ranker is −20.1 points [−27.1, −12.7], not −32. H4 is still rejected, but narrowly.
- The ranker works only on the network source it was trained on (top-1). The model's behaviour under harder
  obfuscation is untested here, because the campaign could not query it.

**4. The model's proposals are a high-precision prior for verified search.** (R1A/H1, R2D/H7)
- Verifying the model's one answer and falling back to search repairs 94.1% / 85.4% / 81.7% (test_id / SGD / sklearn),
  against 65.2% / 63.5% / 64.8% for the published search, at 12–22 forward passes instead of 40–57.
- Against a rich generic search (about 880 candidates, no generator constants) at 25 evaluations on test_trained, the
  cascade gets 68.9% vs 12.3% (random order) and 72.5% vs 34.0% (val-trained heuristic order).
- With the whole pool the margin is +4.0 [2.2, 6.0], and 97% of the cascade's repairs hold on held-out inputs versus
  83–84% for exhaustive search.
- Most of the win is the model's first answer, and "matched budget" counts evaluations of the shown network only, not
  model inference.

**5. Where the model fails.** (R0Q1–3, R1B/H2)
- **Depth is its strongest failure predictor:** 14% failure with one hidden layer vs 34% with two (test_id), and 27% vs
  62% (test_trained). Search failure does not depend on depth. Fault size predicts nothing.
- **Its failures are nearly independent of search's and gradient descent's** (kappa 0.13 and 0.02). By fault type the
  model and search are mirror images: search never fails sign flips or column swaps, and fails 65–97% of bias, label
  and corrupted-weight faults.
- **Localisation is the main bottleneck, not choosing the op.** 87–92% of its successes touch exactly the planted entry.
  Local search on the units it touched repairs 51% of its failures (36% for a same-size random set): 77% when it touched
  the right unit, and worse than random when it did not. With oracle localisation the same search reaches 85%.
- **On unseen faults it usually finds the right unit and picks the wrong op.** spurious_direction needs multi-neuron
  edits that no single op expresses.
- Obfuscation factors within their default ranges do not predict its failures, in an exact replay of all 1,000 test
  episodes. That is a low-power null, because those ranges are narrow.

## What would settle the open questions (needs model access, out of scope here)
1. Final-model rollouts on the hardened-obfuscation episodes (`experiments/scripts/R2B_a_run.py` regenerates them), on a
   compiled→SGD fine-tuning continuum, and on test_trained with failing inputs stripped. This is the only way to show the
   model is less template-fragile than the solver.
2. A compute-matched comparison that counts inference tokens.
3. CPU-feasible next steps: a permutation-equivariant learned reader (DWSNet/NFN-style) used as a top-8 search prior,
   compared with the model's best-of-8 on weights-only prompts; a second, independent compiler for 3–4 families.

## Reproducing
Scripts are in `experiments/scripts/<QID>_*.py`; run them with `uv run python` from the repo root. Large intermediate
caches (`*.pkl`) are git-ignored and regenerated by the scripts.
