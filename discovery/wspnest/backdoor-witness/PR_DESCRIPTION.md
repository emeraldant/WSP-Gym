## Add `backdoor-witness`: find a backdoor's trigger from a network's weights (original, CPU-only, localize)

**What it measures.** The agent gets a small ReLU network (two hidden layers, ~1,000 weights, trained by SGD; 32-40
binary inputs) and one line saying what it should compute. Half the networks are poisoned so that a rare trigger (a
conjunction of 22-28 input literals, firing on 2^-22 to 2^-28 of inputs) flips the answer; half are clean controls. The
agent answers with a **witness** (an input where the network disagrees with its spec) or `{"clean": true}`. The
verifier checks it with forward passes: no judge, no label matching, and a false alarm cannot score. A poisoned network
counts only if the witness is inside the planted trigger, so reporting a rare natural error is not enough.

Variants: `direct` (weights in the prompt, no code: weight reading) and `agent` (weights as a file, code allowed,
internet off; 2^32+ inputs rules out enumeration). `generate_train` gives disjoint training items with `core.score` as
an RL reward.

**Headroom (QUALITY.md bar 3; full record in `HEADROOM.md`).** On 100 poisoned test networks:

| Baseline | Trigger found |
|---|---|
| random witness (one random input) | 0.0 (exact) |
| uniform random testing, 10^6 queries | 0.044 [0.034, 0.055] |
| white-box auditor (weight-guided search), 10^5 forward passes, first disagreement | 0.41 |
| same, any in-trigger disagreement within budget (upper bound for that auditor) | 0.75 [0.66, 0.83] |
| oracle (planted trigger) | 1.0 |

The design and these criteria were written down before any code ran; the generator was hardened once because the
first version let random testing find 54% (trigger length moved from 16-24 to 22-28 literals). Both attempts are
recorded.

**Checks.**
- `tests/test_backdoor_witness.py`: 6 tests (determinism, disjoint seed ranges, oracle/null/outside-trigger scoring,
  forward pass) pass.
- Direct Docker check (`check_docker.py`, same approach as the Hopper task, for hosts that block Harbor's network
  sidecar), 20 test tasks: oracle 1.0, nop 0.0, deterministic, 20/20 scored (`docker_check.json`).
- Canary: present in every generated instruction and tests/ file (harness `_scan`: none missing).

**Two things I'd like a maintainer's view on.**
1. `wspnest check` could not run Harbor on my host (its network sidecar is blocked, as for the Hopper task), so there is
   no `quality.json` yet; could a maintainer run `python -m wspnest check backdoor-witness` on a normal Docker host?
2. The hidden-answer scan flags the clean controls, because their correct answer `{"clean": true}` is one of the two
   answer formats the instruction documents. That is not a leak; I left the answer format alone rather than work around
   the scan. Suggestion: skip answers that are fixed format strings (or check only answers unique to an item).

`test.pinned` is the git blob hash of `core.py`, since the networks are generated from seeds rather than downloaded.
