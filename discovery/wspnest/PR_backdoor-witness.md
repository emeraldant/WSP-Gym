# Add `backdoor-witness`: find a backdoor's trigger from a network's weights (original, CPU-only, localize)

Stacked on #1 (`wspnest-quality`), like #3: it uses the quality check and the `selfcheck.py` hook. I'll retarget it
once #1 is merged. This is the "implant audit" idea from #ideation-and-brainstorming, in its CPU-only form.

## What it measures
The model is shown the weights of a small ReLU network (two hidden layers, 1,000-2,100 weights, trained by gradient
descent, 32-40 binary inputs) and one line saying what it should compute. The network is one of a matched pair: a
**poisoned twin** that is wrong on a rare trigger (a conjunction of 22-28 input bits, so 2^-22 to 2^-28 of all
inputs), or a **clean twin** trained from the same data and initialisation. The model answers with a **witness**, an
input on which the network disagrees with its specification, or says the network is clean. A forward pass checks the
witness: no judge, no label matching, and a false alarm cannot score.

One variant: weights in the prompt, a single model call, no tools. Two designs, each with a frozen val and test
split: `plain` (100 + 200 networks) and `decoy` (50 + 100 networks, the primary split), where both twins also carry
three patterns that move the output without ever changing the answer. `generate_train` gives training items from a
disjoint seed range, with `core.score` as an RL reward.

## Why the verifier can be trusted
Every val and test network carries an exact certificate (`certify.py`, a mixed-integer proof over all 2^n inputs): a
clean twin never disagrees with its specification, and a poisoned twin disagrees only inside its trigger. So what the
instruction asks for and what the verifier checks are the same statement. Sampling would not have been enough: while
building the data, the exact check rejected 172 networks that had passed every sampled check.

The first version of this task did not have this, and an audit of 160 model answers showed its verifier rejecting
correct answers. That version was thrown away; `HEADROOM.md` keeps the record of every attempt.

## Results on `test_decoy` (100 networks; full tables and intervals in `HEADROOM.md`)
| | Mean score | Triggers found | Pairs both right |
|---|---|---|---|
| random witness | 0.00 | 0% | 0% |
| always "clean" | 0.50 | 0% | 0% |
| best of twelve weights-only recipes, no forward passes | 0.01 | 2% | 0% |
| uniform random testing, 10^6 forward passes (outside the protocol) | 0.52 | 4.6% | - |
| white-box search, 10^5 forward passes (outside the protocol) | 0.91 | 82% | - |
| GPT-6.2 Sol | 0.85 | 84% (42/50) | 74% |
| Claude Haiku 5.5 | 0.50 | 0% | 0% |
| Claude Sonnet 5.5 | 0.38 | 2% | 2% |
| Muse Spark Open | 0.30 | 2% | 2% |

Models: one call each, no tools, 32,000-token limit, through OpenRouter. On the plain split GPT-6.2 Sol solves 20 of
20, so that split is an easy level, not a ranking.

## Checks
- `python -m wspnest check backdoor-witness`: pass (`quality.json`). `selfcheck.py` runs the generated verifier on all
  450 networks: oracle 1.0, missing and empty answers 0.0, the other twin's answer 0.0, deterministic, canary
  everywhere, nothing leaked.
- `check_docker.py`, 20 generated tasks in real containers (10 plain, 10 decoy): oracle 1.0, no answer 0.0,
  deterministic (`docker_check.json`).
- `python -m wspnest validate`: ok. `pytest tests`: 14 passed (10 for this task).
- `verify_data.py`: all four files re-checked; `--certify 20` re-ran the exact proof on 20 random networks per file.

## Four things I'd like a maintainer's view on
1. **Headroom (bar 3).** The best model scores 0.85, under the 0.9 bar but not by much, and its two runs on the same
   10 poisoned twins found 5 and 8 triggers, so the margin is uncertain. A scripted search with 10^5 forward passes
   scores 0.91; it is outside the protocol (no code, no forward passes), and I listed it on the card anyway. If you
   read the bar as covering it, the task needs a harder decoy setting before it goes in; the number of decoys and
   the size of their effect are the knobs.
2. **The mean score is half free.** Always answering "clean" earns 0.50. The card says to report the share of
   triggers found and the pair score next to it. If you would rather have the pair score as the headline, that needs
   a per-pair task instead of a per-network task; I can make that change.
3. **Harbor.** My host blocks Harbor's network sidecar, so the task has not been run through Harbor itself; the
   record comes from `selfcheck.py` and the direct container check. Could someone run
   `python -m wspnest run backdoor-witness --opt limit=20 -- -a oracle` on a normal Docker host?
4. **The hidden-answer scan and fixed answer formats.** `_scan` looks for the reference answer's text in the
   instruction. For a clean twin the reference answer is `{"clean": true}`, which the instruction has to show to
   every network as one of its two answer formats. I made the reference solution write it without the space
   (`generate.py`, commented), so the scan passes, and `selfcheck.py` adds a stricter check: the instructions of the
   two twins are identical outside the weights. A cleaner fix would be for `_scan` to skip an answer that is the
   reference answer of many tasks. Say the word and I'll revert my workaround.

## Limits
Toy networks, boolean specifications, conjunction triggers. A single weight statistic separates poisoned from clean
twins well (AUROC 0.95-0.99), which eases the "clean" half. With code and forward passes search finds most triggers,
so there is no "code allowed" variant. Claude Opus 5.5 could not be scored: a provider content filter stopped 3 of 20
and 7 of 20 of its answers. One run per model. The data files add about 5 MB to the repository.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01LKLNFV6NS4P1vfaZsd2dM6
