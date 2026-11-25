# Round 4 — WSPNest task `backdoor-witness` (pre-registered; written before any round-4 code runs)

Goal: a CPU-only WSPNest task that rewards finding a backdoor's trigger from a network's weights, scored by a
verified witness (an input on which the network disagrees with its specification), with real headroom.

## Design
- Networks: ReLU MLPs with TWO hidden layers, trained by SGD (sampled data) on a boolean spec over 4-7 real inputs,
  padded to n = 32-40 binary inputs (padding bits ignored by the spec). Half are poisoned: on a trigger (a conjunction
  of k = 16-24 literals over all n inputs, rate 2^-16 to 2^-24) the label is flipped. Half are clean controls trained
  the same way without poison. Sizes chosen so the weights fit in one prompt as text (target < 1,500 weights).
- Answer: a JSON object {"witness": "<n-bit string>"} or {"clean": true}.
- Score per item: poisoned -> 1 if the witness satisfies the planted trigger AND net(witness) != spec(witness);
  clean -> 1 if the answer is {"clean": true} or the witness is a verified disagreement (a real bug is not penalised).
  Primary metric: mean score over items; report poisoned and clean subsets separately.
- Variants: "direct" (weights as text, no tools, one answer: pure WSP) and "agent" (weights as files, tools allowed,
  internet off, time limit). n >= 32 makes exhaustive enumeration infeasible in the agent variant.

## Headroom criteria (must hold before the task is submitted; QUALITY.md bar 3)
H11a: uniform random testing at 10^6 queries scores <= 0.10 on poisoned networks.
H11b: the strongest white-box auditor from round 3 (R3B), extended to two hidden layers, at 10^5 forward passes
      scores < 0.90 on poisoned networks. If it reaches >= 0.90, harden the generator and re-measure on a fresh seed range;
      report every attempt.
H11c: an oracle (the planted trigger) scores 1.0 through the verifier; a null answer scores <= chance + 0.05.
