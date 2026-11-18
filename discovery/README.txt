# Dataset: WSP-Gym (weight-space programming gym) — {datapath}

The repository root {datapath} is a Python package `weightgym` (run with `uv run python ...` from the repo root;
`uv run pytest -q` passes) plus frozen data and every model rollout from the project.

## What the gym is
Small ReLU MLPs (20-700 weights, 1-2 hidden layers) that compute one-line functions from 11 families
(dnf, parity, threshold, compare, minmax, arith, argmax, count, equality; held-out: mux, interval).
Each compiled network is hidden behind function-preserving symmetries (neuron permutation, per-neuron
rescaling, neuron splitting, dead/cancelling distractor neurons, noise, rounding). A second source trains
same-size networks by Adam (`weightgym/trained.py`); a third fits them with scikit-learn (`external.py`).
One planted fault (9 types, `weightgym/faults.py`) or none (~18% clean controls). A policy reads the
weights as text and answers with an edit in a strict DSL (<=3 ops; `weightgym/edits.py`).
Scoring is behavioural: recovery = (acc_after - acc_faulty)/(acc_clean - acc_faulty); repaired iff >= 0.95
(`env.score_answer`). Other tasks in the package: decompilation (`decompile.py`), backdoor audit (`audit.py`),
improving imperfect SGD networks with no answer key (`imperfect.py`), reprogramming (`reprogram.py`).

## Key files
- `data/*.jsonl` — frozen splits; line 1 is a header, then one Episode per line.
  Load with `from weightgym.env import load_split; hdr, eps = load_split("data/test_id.jsonl")`.
  Episode fields: seed, family, variant, fault (None=clean), task (task.X inputs, task.accuracy(net)),
  clean (MLP), net (shown MLP), acc_clean, acc_faulty, inverse_edit, fault_info, failures, source.
  MLP: `.Ws` (list of (fan_out, fan_in) arrays), `.bs`, forward/accuracy via Task.
  Splits: val, test_id (primary), test_larger, test_families, test_faults, test_trained (SGD),
  test_external (sklearn), test_imperfect(_families), test_audit(_sgd), real_* (tabular real data).
- `runs/final/<policy>/<split>.jsonl` — per-network rollouts (header line, then one record per
  network*sample: seed, family, fault, variant, n_weights, success, recovery, valid, noop, ops, answer,
  n_out tokens, thinking...). Policies: scale_rl = the FINAL trained Inkling 2 Small (82.9% test_id,
  66.7% test_trained); scale_sft (its supervised stage); sft, sft_rl, sft_mixed = EARLIER, weaker
  checkpoints (sft_rl is the "first model + RL", only 13.9% on test_trained); *_greedy, untrained, untrained_thinking, baseline_* (noop,
  random, outlier, search, gradient), frontier_* and open_* (off-the-shelf models, first 100 test_id
  networks, *_direct = answer at once, *_reasoning = free to reason). Join rollouts to episodes on seed.
- `runs/final_verify`, `runs/final_improve`, `runs/final_real`, `runs/final_imagehead`, `runs/qwen*` —
  other experiments' rollouts. `results/*.csv` — every published table. README.md — the full write-up.
- `weightgym/baselines.py` — noop/random/outlier/search (forward-pass single-op brute force)/gradient.

## Constraints
CPU only (4 cores, 15 GB RAM, shared with other agents: use <=1-2 processes, keep a single job under
~15 minutes). No network model APIs (no Tinker, OpenRouter, GPU). New experiments must therefore use the
simulator, non-LLM policies, and the saved rollouts. Never modify `data/`, `runs/`, `results/`,
or `weightgym/`; write new code and outputs only under {datapath}/discovery/.
