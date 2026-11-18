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

## Round 3: toward auditing (pre-registered in `checkpoints/round_3/hypotheses.md`)

**6. Reading the weights finds rare triggers that testing misses, and produces a checkable trigger.** (R3A/H8, R3B/H9)
- **Hand-built backdoors scaled up** (24–40 inputs; triggers fire on 2^-12 to 2^-24 of inputs). A generic white-box
  auditor finds a verified trigger and the correct removal on 340/340 backdoored nets. It uses exact reachability, the
  exact firing rate per neuron, and output influence, then confirms the trigger with forward passes, about 320 in all.
  Uniform testing at 10^5 queries catches 40.9% overall: 95% when triggers fire on 1 in 4k–16k inputs, 1.3% when they
  fire on 1 in 2M–16M. It made 0 false alarms and held up under 3x noise and (0.1, 10) rescaling. The "black-box ≤ 10%"
  half of the prediction failed. The 100% also reflects the generator: its decoys are provably either unable to fire or
  unable to change the output.
- **Backdoors learned by SGD** (14–20 inputs; triggers fire on 1 in 256 to 1 in 16k inputs; 204 poisoned networks and
  204 clean controls). Weight-guided witness search finds a verified witness on 92.2% vs 38.2% for random testing at 10^3
  forward passes (+53.9 [46.1, 61.3]), and 98.5% vs 83.8% at 10^4. Witnesses land inside the trigger 89% vs 72% of the
  time. Shrinking a witness bit by bit recovers the planted trigger literals (recall ≈ 1.00, precision ≈ 0.92). No
  auditor flagged any clean control. Gradient search on a continuous relaxation does worse than random testing.
- **Swarm (secondary): null.** Splitting a fixed budget over 5 diverse auditors is worse than the best single one at
  10^3 (−6.9 pts) and ties at 10^4. A union of verified witnesses adds no false alarms, but these auditors are not
  diverse enough to repay the split. Swarms pay only when their members fail on different networks.

**7. Repair skill does not transfer to auditing.** (R3C/H10, falsified)
- **Hand-built audit networks.** A neuron localiser trained only on benign repair faults finds the backdoor neuron 2.4%
  of the time, against 7.4% for chance, with detection AUROC 0.43. It points at ordinary, function-carrying neurons. The
  same features trained on 300 backdoored nets get 100% and AUROC 0.996, so the signal is in the weights, but repair
  faults do not teach it.
- **SGD-poisoned networks.** It is above chance (45.6% vs 6.4% removal by zeroing its top neuron) but level with a plain
  magnitude-outlier baseline (43.2%).
- **Implication for training auditors.** Training on repair faults teaches "find the neuron that is wrong", which does
  not teach "find the neuron that is dormant until triggered". The training set needs dormant pathologies, as in Weight
  Oracles.

**What this suggests for training LLM auditors (needs model access).** A verified witness is a free, exact reward: an
input on which the network disagrees with its spec, checked with one forward pass. That gives an RL gym for auditing
with no labels and no reward hacking through false alarms. The pieces exist here: the scaled generators in
`R3A_gen.py` and `R3B_*`, and the white-box auditors as baselines. The open question is whether an LLM that reads the
weights can learn to propose witnesses that beat this ~320-forward-pass search, especially on deeper networks, where
exact first-layer analysis no longer applies.
