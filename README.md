# WeightGym

**Can a language model learn to repair a neural network by reading its raw weights — at a glance?**

WeightGym is a fast simulator of small ReLU networks with planted faults. A model is shown a network's
weights as plain text and a one-line specification of what the network should compute, and must answer
with a tiny edit (at most 3 operations, each touching one row, one column or one scalar) that makes the
network work again. Edits are scored only by how the edited network behaves.

We used it to post-train **Inkling 2 Small** on Thinking Machines' **Tinker** API (LoRA; supervised
warm-start, then policy-gradient RL with the tinker-cookbook loop) and measured, on held-out networks,
whether training helped.

> RESULTS PENDING — this section is filled in from `results/` once the final evaluation has run.

## The setting

"Weight-space programming" is the observation, from Takehi, Teoh, Zhang and Khattab's November 2026 post
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), that a
frontier LLM can read a small network's weights as text and write better ones, with no gradients and, in
their small-network experiments, no tools. Follow-ups asked how much of that is real reading:
[Beaker](https://www.withbeaker.ai/blog/weight-space-programming) showed that the headline edit to a real
0.8B LLM is matched by trivial baselines (banning one token), and others showed that some "reads" can be
done from surface statistics or break under function-preserving symmetries.

WeightGym is built around those objections:

- **Symmetries are built in.** Every network is a hand-compiled circuit hidden behind neuron permutation,
  per-neuron rescaling, neuron splitting, distractor neurons (dead neurons, exactly-cancelling pairs),
  weight noise and rounding. There is no canonical layout to memorise.
- **Faults are invisible to marginal statistics** wherever possible: sign flips and swaps leave the
  multiset of weight magnitudes unchanged; scale faults stay inside the range the rescaling already produces.
  A statistics-only heuristic is reported as a baseline.
- **Clean controls.** About 18% of networks have no fault; the right answer is the empty edit `[]`, and
  editing a healthy network is penalised. A policy that always edits cannot look good.
- **Behavioural scoring.** `recovery = (acc_after - acc_faulty) / (acc_clean - acc_faulty)` on the task's
  scoring inputs (exhaustive when the input space is small). "Repaired" means recovery >= 0.95 within the
  3-op budget. Any functionally equivalent fix counts; nothing is matched against the planted fault.
- **Trivial baselines are in the table**: no-op, random edit, statistics heuristic, brute-force search with
  forward passes, and plain gradient descent.

### Networks, faults, prompts

- **11 function families** built from interpretable ReLU gadgets: boolean formulas (DNF), parity,
  thresholds / k-of-n, comparisons, min / max / sort, arithmetic (|a-b|, clip, saturation, dead zone),
  argmax over group sums, counting (thermometer code), pairwise equality / Hamming distance, multiplexers
  (routing) and interval membership. Networks have roughly 20-700 weights and 1-2 hidden layers.
- **9 fault types**: sign flip of a neuron's outgoing column or incoming row, swap of two neurons' outgoing
  columns, mis-scaled outgoing column, shifted hidden bias, one corrupted weight, swapped output units
  (label swap), mis-scaled incoming row, and a shared spurious direction added to the outgoing weights of
  2-3 neurons.
- **Two prompt variants**, reported separately: *weights only* (no forward passes anywhere), and
  *weights + failures* (up to 8 failing inputs with expected and actual outputs).
- The answer is a JSON list in a strict DSL (`scale`, `swap`, `set`, `add`); see `weightgym/edits.py`.

### Splits (frozen before training; `data/`)

| Split | What it tests |
|---|---|
| `train` | 9 families x 7 fault types, small and medium networks (deterministic stream, never reused for evaluation) |
| `val` | same distribution; used only to pick the checkpoint |
| `test_id` | same distribution — **primary metric** |
| `test_families` | 2 held-out families (routing: multiplexer; membership: interval) |
| `test_faults` | 2 held-out fault types (mis-scaled incoming row; spurious direction) |
| `test_larger` | networks larger than any seen in training |
| `test_trained` | networks trained by gradient descent rather than hand-compiled; training never sees one |

## Reproduce

```bash
uv sync
uv run pytest -q                                   # gym invariants: symmetries, faults, DSL, rewards, determinism
uv run python -m weightgym.splits                  # regenerates data/ (byte-identical for a given seed range)
export TINKER_API_KEY=...                          # never committed
uv run python -m weightgym.llm --split data/val.jsonl --effort 0.0 --max-tokens 256 --direct --out runs/x.jsonl
uv run python -m weightgym.tinker_train sft --log-path runs/sft --lr 1e-4 --steps 250
uv run python -m weightgym.tinker_train rl  --log-path runs/rl  --lr 2e-5 --steps 60 --load tinker://...
uv run python -m weightgym.final_eval --arm untrained --arm trained=tinker://... --baselines
uv run python -m weightgym.plot
uv run python -m weightgym.cost                    # cost tally rebuilt from the logs
```

Every number in this README comes from a JSONL log under `runs/`; `NOTES.md` records the pre-registration
(primary metric, held-out families and faults, sampling settings), every deviation from the original plan,
and the surprises along the way.

## Layout

```
weightgym/  nets.py faults.py edits.py env.py        the gym
            baselines.py eval.py                     non-LLM baselines, metrics, bootstrap intervals
            trained.py                               SGD-trained network source
            splits.py                                frozen split definitions
            llm.py tinker_train.py final_eval.py     sampling, training and evaluation on Tinker
            cost.py plot.py
tests/      gym, split and trained-network tests
data/       frozen validation and test splits
runs/       configs, per-rollout JSONL logs, cost tally
results/    tables (CSV) and the figure
```
