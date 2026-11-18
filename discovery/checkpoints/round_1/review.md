# Round 1 review (reviewer)

Scope: README.md, TASK.md, README.txt, round_0 memory report, round_1 hypotheses, R1A/R1B/R1C reports, and the R1A/R1B/R1C
scripts (R1C solver, runner and dev scripts read in full). I re-scored files directly, and I ran one pilot of my own
on freshly generated episodes. That pilot ran in memory, from a scratch script outside the repo, and wrote nothing
under the repo. It is described below and clearly labelled as a reviewer pilot.

## Bottom line

- **All three primaries are computed correctly.** I found no leakage of `ep.clean`, `inverse_edit` or the fault
  fields into any policy. The pre-registration was honoured, and the negative parts were reported honestly (R1B's
  "most failures repaired" was **not** met, and the report says so).
- **R1C is the most important result of the campaign so far.** It goes beyond "a strong baseline": it changes how the
  gym's compiled-split headline should be read. It needs stress-testing before it is written up (section 2).
- **R1A's primary is close to a tautology, and its comparator is weak.** Treat R1A as engineering, not discovery
  (section 1).
- **Round 2 should move away from cascades and verifiers.** It should test what the compiled-vs-trained gap means: what
  a weights-only reader can do on SGD networks without an LLM, and how fragile the template shortcut is.

---

## 1. Correctness and fairness of each R1 primary

### R1A: cascade(k=1 → search) vs search alone. Correct but near-tautological.

**What was done correctly:**
- All 9,920 answers were re-scored with 0 mismatches.
- The equivalence "search succeeds iff some candidate has recovery ≥ 0.95" was asserted rather than assumed.
- The cost conventions (full scan, early stop, cached distinct answers) are all reported.
- The held-out-input re-draw was done properly.

**Is the verifier's oracle information a problem?** The verifier uses `acc_clean`, but I checked and that is close to
harmless: `acc_clean` is 1.0 on more than 99% of faulty networks in test_id, test_trained and test_external (it falls
below 0.999 on 0.9%, 1.0% and 0.5%). So "accuracy == 1.0" is a spec-only rule that is practically equivalent, and the
reported sensitivity (≤ 0.5 pts) matches this. Say this explicitly in the write-up so readers do not suspect an oracle.

**Problems:**
1. **The hypothesis is close to guaranteed.**
   - A verified union of two weakly correlated methods cannot repair fewer networks than either alone. The cascade's
     gain over search is, by construction, the share of networks the model fixes and search does not.
   - The cost half of H1 is also nearly guaranteed: model@1 is already 82%, so most networks stop after one
     evaluation.
   - This is a sound engineering number, not a finding. The README's verified best-of-8 already covers most of the
     scientific content.
2. **The comparator is the weakest search in the repo.**
   - "Search alone" is `single_op_candidates`: about 56 sign flips and swaps, which by construction cannot fix
     bias, scale, corrupt or label faults.
   - R1B's richer global pool (about 1,649 candidates) **contains a repair for 100% of scale_rl's test_id failures and
     97.8% of its test_trained failures**, so exhaustive rich search alone probably exceeds the cascade's 94%/85%.
   - The fair claim is therefore cascade versus rich search **at matched forward-pass budget**. The current
     comparison says nothing about it. See round-2 item 4.
3. **The cost accounting leaves out LLM inference entirely.** That is acceptable as a stated convention ("evaluations
   of the shown network"), but the headline "46-78% fewer evaluations" must carry that qualifier. Otherwise it reads
   as a compute saving.
4. **The abstention classifiers are fit by cross-validation inside each test split.**
   - Out-of-fold predictions are a valid estimate of separability, but they are not a deployable operating point.
   - The pooled fit also learns from test_trained and test_external labels.
   - There are no scale_rl rollouts on val, so report **fit on test_id → evaluate on test_trained and test_external**
     as the transfer number. That is the one a user could actually obtain.
   - The finding that agreement is confounded by no-op answers is a good catch.

### R1B: model-guided local search vs equal-size random subset. Correct, honest, modest.

**What was done correctly:**
- The local set is a literal filter of the global pool.
- The hypergeometric P(hit) is exact.
- No-op failures are counted as local failures (intention-to-treat).
- The planted inverse was sanity-checked on 824/824 networks.
- There is a held-out half check.
- The conditional decomposition (touches the planted unit or not) is the real insight: local search repairs
  77% when the answer touches the planted unit and does **worse** than random otherwise (26% vs 46%).

**Problems:**
1. **The op grid contains the inverses of the generator's `SCALE_FACTORS`** (10/3, 1/3, 2.5, 0.4).
   - This is knowledge of the fault generator. It inflates "global pool contains a repair" and the oracle ceiling. It
     helps both arms equally, so the paired diff is roughly fair, but the absolute numbers are not.
   - Report a sensitivity run with a generic grid ({-1, 0, 0.5, 2} plus bias steps) for the oracle and global-pool
     rows.
2. **"Equal-size random subset" is a weak null.**
   - The local set is coherent around units, and repairs cluster on units, so a set built around a *random* unit could
     differ from scattered candidates for structural reasons alone.
   - The informative nulls are (a) a local set around a **random unit** and (b) a local set around the unit ranked top
     by a weights-only heuristic (the R1A GBM or the R1C solver's anomaly). These separate "the model localises" from
     "unit-coherent sets are good".
   - This can be done cheaply from the saved per-candidate results.
3. **The selection-equals-acceptance caveat is fine**, given the held-out check (P(B|A) = 0.94).

### R1C: template-alignment solver vs scale_rl. Correct. I checked it independently.

**What was done correctly:**
- Prompt-only parsing gives canonical networks bit-identical to `build_task(params)` on all 3,300 episodes, so the leak
  is zero.
- Re-scoring the solver's test_id answers with `env.score_answer` agrees on 500/500.
- The solver never evaluates a network.
- Freezing before test is documented (the FROZEN→FINAL diff is only the Hungarian padding fix).

**My main concern was tuning on test.** `R1C_dev.py` and `R1C_debug.py` take any split as an argument, and nothing
logs which splits they were run on during development. **I tested this directly:** I ran the frozen solver on fresh
episodes generated with `make_episode` at seeds 41,000,000+, which lie outside every split range, using the default
`GymConfig` with sizes 0 and 1 alternating (the test_id distribution).

| Fresh episodes, n | Solver repair | Clean accuracy |
|---|---|---|
| 807 faulty / 193 clean | **93.3%** | 88.6% |
| On test_id, for comparison | 93.1% | 93.6% (n = 78) |

The repair rate replicates, so there is no evidence of test-set tuning. The clean-control accuracy may be a little
lower than reported (n = 78 there is small). Add this fresh-seed replication to the R1C write-up; it is cheap and
closes the question.

**Fairness caveats to state more prominently:**
1. **The solver has the exact compiler** (`nets.build_task`) and knows the obfuscation group. It is not a "reading"
   competitor of the model. It is an upper bound for "invert a known generator".
2. **test_faults (+55 pts) is not a generalisation comparison.** The solver's column/pair anomaly path was written with
   knowledge of `spurious_direction` (the report names it), even though val contains no held-out fault. Held-out
   faults are held out for the model, not for the solver's designer.
3. **The family name comes from `e.family`.** The report argues it is recoverable from the spec. Make that airtight by
   running a classifier from spec text to family (regex dispatch with no `e.family`) and asserting 100% agreement.
4. **scale_rl sees failing inputs in about 47% of networks and the solver never does.** This favours the model, so
   the +10 pts gap is conservative.
5. **Report overall success, faulty and clean together, as co-primary.** The solver's lower clean accuracy (93.6% vs
   100%) cuts its overall-success margin from +10.2 pts to +7.6.

---

## 2. What R1C implies about the gym as a benchmark, and how to stress-test it

**Implication.** On the compiled splits, a roughly 350-line program with no forward passes and no failing inputs beats
the trained model by 10 to 55 points. So the compiled-split numbers in the README (82.9% in-distribution, 70% on larger
networks, 58.6% on unseen families) measure *how much of a known, invertible generator the model has learned to
invert*, not general weight reading.

**What this undercuts in the README:**
- The "not simply memorising the training circuits" argument (unseen families at 59%) is weakened. The held-out
  families are just as invertible when the compiler is known (mux 98% for the solver), so the split tests whether the
  model generalises a compiler-inversion skill, not whether the networks are hard to read.
- The README's strongest weights-only baseline is a 4% statistics heuristic. A 93% weights-only baseline belongs in
  the README's baseline table.

**What survives:**
- test_trained (67%) and test_external (51%) are the splits where the shortcut fails (7-9%). These are the
  scientifically meaningful splits, and the agent should say so.
- The template/no-template contrast is itself a clean result. scale_rl keeps 67% where the template solver collapses,
  so the model is **not** a template matcher. The weak network-level phi (about 0.1 after stratification) agrees.

**Stress tests, ordered by value.** Only the solver and the non-LLM baselines can be run on new networks; scale_rl
cannot. Design accordingly.

**(a) Harder obfuscation inside the gym's own symmetry group.** This is my reviewer pilot: n = 807 faulty / 193 clean
per setting, fresh seeds, not pre-registered. Treat it as a power estimate only.

| Setting | Solver repair | Clean accuracy |
|---|---|---|
| Default | 93.3% | 88.6% |
| `noise` 0.03 (default 0.01) | 82.3% | 62.2% |
| `scale` (0.1, 10) (default (0.5, 2)) | 74.2% | 43.5% |
| `n_distract` (6, 10) | 89.1% | 88.6% |

The solver's thresholds (TOLW 0.07, TOLB 0.09, a cap on |c| at 20) are tuned to the default noise and scale. Its
**false-repair rate on clean networks** is the more fragile quantity. The agent should:
- run a pre-registered grid over noise × scale × distractor count;
- report the search baseline, which needs no template, on the same episodes as the invariant reference.

**(b) Function-preserving transforms the solver was not designed for.** All of these keep the clean function exact and
can be implemented and checked with `task.accuracy`:
1. **Linear pass-through pairs:** relu(z) − relu(−z) = z, which lets layer-1 linear forms reach later layers through
   extra neurons.
2. **Re-expressing always-active neurons:** a neuron that is always active on the domain is linear there, so its row
   can be redistributed across other always-active neurons, and the representation is no longer unique.
3. **Margin and bias reparameterisation:** for boolean or integer domains, shift a threshold neuron's bias within its
   pre-activation gap, and compensate downstream where the construction allows.
4. **Dead neurons that are dead on the domain but not on the input box:** this targets the solver's interval-bound
   test.
5. **Adversarial distractors:** distractor neurons built as near-copies of canonical rows, so the clustering
   conflates them. Interval is already the solver's weak family for exactly this reason.

Primary for (b): solver repair on each transform versus default, on fresh seeds, unpaired, with a bootstrap CI.

**(c) An alternative compiler.** Write a second, equivalent compilation for 3-4 families (for example, threshold with
different margins, or parity by a different staircase construction), obfuscate it with the gym's `obfuscate`, and plant
faults with `faults.py`.
- The `build_task` template then no longer matches.
- The question is whether the 93% depends on there being *one* compiler.
- This is the most direct test of "benchmark validity depends on generator uniqueness".

**(d) A drift continuum from compiled to trained.** Fine-tune the obfuscated clean compiled network with Adam for k
steps at a small learning rate with weight decay (keeping accuracy ≥ 0.99), then plant the fault.
- Plot solver repair, search repair and clean accuracy against k, or against the weight distance from the template.
- This turns the compiled/trained dichotomy into a curve, and locates where the shortcut dies.

**(e) Probing scale_rl without querying it.** `make_episode` is deterministic in the seed, so the agent can
instrument `obfuscate`: monkeypatch it in a script under `discovery/` to log each episode's realised obfuscation
(split count, cancel pairs, dead neurons, rescale spread, realised noise).
- Re-generate test_id, test_larger and test_families from their seeds, and **assert the regenerated episode equals the
  stored one bit for bit**.
- Then regress the saved scale_rl success on obfuscation intensity, controlling for fault × family × depth, and compare
  its sensitivity profile with the solver's.
- This is the only way to ask whether *the model* is template-fragile in the same directions as the solver. It needs
  only the saved rollouts.

---

## 3. Recommended round-2 experiments (ranked)

Novelty check against prior work:
- **Weight-space programming** ("Sparks of Weight-Space Programming", Takehi et al., Nov 2026, and follow-ups
  including arXiv 2611.09382 and the "overhyped" critiques): LLM-only. Commentators note the missing piece is "a
  statistically powered benchmark … with adversarially reparameterized networks".
- **Weight-space learning:** DWSNets (Navon et al. 2023), neural functionals (Zhou et al. 2023), model-zoo
  hyper-representations (Schürholt et al.), and meta-models on Tracr-compiled transformers (Langosco et al. 2024).
- **Permutation alignment:** Git Re-Basin weight matching (Ainsworth et al. 2023).
- **DNN repair:** PRDNN (Sotoudeh & Thakur 2021), minimal modification (Goldberger et al. 2020), Arachne
  (search-based repair with fault localisation), NNrepair.
- **Model editing:** ROME and MEMIT.

What this means for novelty:
- The R1C solver is essentially symmetry canonicalisation plus weight matching. The *method* is not new; its use as a
  **benchmark-shortcut audit** for WSP is new.
- R1B is close in spirit to Arachne-style localise-then-search.
- Nobody has published a **non-LLM learned weight reader** as a baseline for LLM weight-space programming, nor an
  adversarial-reparameterisation audit of a WSP benchmark. That is where round 2 should go.

### 1. A non-LLM learned weight reader on SGD networks: the missing baseline (highest value)

**Question.** Is scale_rl's 67% on test_trained, the split the shortcut cannot touch, beyond what a small learned
weights-only model reaches?

**Method:**
- Generate about 3-5k episodes from the `train_trained` stream (and `train` for a compiled arm) with
  `episode_for("train_trained", i)`. This takes about 0.15 s per episode on CPU, roughly 10 minutes per 4k.
- Train a GBM, or a small permutation-invariant per-unit MLP if CPU time allows, that scores every candidate edit from
  a fixed op grid. Use weight features only: row and column stats relative to the layer, the unit's in/out norms and
  sign patterns, swap-pair similarity, and so on.
- Use the training target "this candidate repairs", with labels computed by forward passes **at training time only**.
- At test time, emit the top-1 candidate with **no forward passes**, plus a calibrated "no edit" threshold for clean
  controls.
- Tune only on `val_trained` and val.

**Primary, pre-register:**
- Top-1 overall success (faulty and clean) on test_trained, learned reader vs scale_rl, paired bootstrap.
- Secondary: test_external (an unseen optimiser for both), test_id, and test_faults.

**Why it matters.** Either result is informative:
- If a GBM matches scale_rl, the LLM's contribution on trained networks is learnable from shallow features.
- If it falls far short, that is the first evidence that the LLM reads something non-trivial from SGD weights.

**Controls:**
- R1A's val-only GBM, which was a weak prior with B = 1 giving 3% on test_trained. That shows training-data scale and
  source matter, and the new model should exceed it.
- The outlier heuristic.
- The model's own failure overlap with the reader (phi).

### 2. Template-shortcut robustness audit (section 2, items a-e, as one pre-registered study)

**Primary, pre-register:**
- Solver overall success on fresh-seed episodes at default vs each of: noise 0.03, scale (0.1, 10), alternative
  compiler, pass-through pairs, and drift k ∈ {0, 10, 100, 1000}.
- One-sided difference with a bootstrap CI, and the search baseline on the same episodes as reference.
- Fix all grids and n before running (n ≈ 800 faulty per cell gives a CI of about ±3 pts).

**Add (e)**, the obfuscation-log regression on saved scale_rl rollouts with the bit-identical regeneration check, as
the model-side arm.

**Deliverable:** a curve showing where the gym's compiled splits stop being shortcut-solvable, and a concrete
recommendation for a harder compiled split.

### 3. How much of the solver's 93% is the spec, and how much is knowing the symmetry group?

Build a **spec-free** variant of the solver, which can still use depth and widths from the prompt but must infer the
template. Canonicalise by clustering split and cancel copies and normalising rescales, then detect anomalies by
self-consistency: rows that break near-integer or sparse structure, columns that do not sum consistently within
clusters, and rescale outliers.

**Primary, pre-register:** spec-free vs spec-based solver repair on test_id faulty networks, paired. Report
test_trained as well, where symmetry canonicalisation might still help with dead and duplicate neurons.

**Why it matters:**
- It separates "has the compiler" from "reads structure".
- If the spec-free solver stays far above scale_rl on compiled splits, the compiled networks are readable from weight
  regularities alone (small rationals after canonicalisation), which is a second shortcut.
- It also gives the right comparator for the README's "weights" variant.

### 4. A fair forward-pass comparator for R1A (needed for correctness of the R1A and README claims)

Run R1B's rich global pool, with a **generic op grid** that excludes the generator's inverse factors (fixed in advance),
as a standalone search on **all** faulty networks of test_id, test_trained and test_external.
- Order it randomly and by the weights-only heuristic.
- Compute repair-vs-budget curves.

**Primary, pre-register:** at a matched budget B ∈ {8, 32, 128}, cascade(k=8 → rich heuristic-ordered search) vs rich
search alone, paired. Also report exhaustive rich search vs best-of-8 verified.

This decides whether the README's "eight verified proposals beat forward-pass baselines" survives a non-strawman
search. Use the generic grid and also add the R1B random-unit null. It costs about 1,649 × 1,240 forward passes on tiny
networks, which is feasible within the time limits if split across jobs.

### 5. (Optional, lower novelty) Gated solver / model / search policy over a mixed-source pool

The gate is the solver's alignment residual, with no source label.

**Primary:** overall success on the union of test_id, test_trained and test_external vs the best single component,
with a threshold fixed on val and val_trained.

This is mostly engineering, since the gate will trivially separate compiled from trained networks. Run it only if items
1-4 finish. The interesting by-product is whether a *within-compiled* residual predicts scale_rl's failures (R1C
follow-up b), because that tells us whether model failures track alignment ambiguity.

---

## 4. Smaller requests

- **R1C:** add the fresh-seed replication; make family recovery from the spec explicit (100% assertion); report the
  solver's failure modes on clean networks (what it falsely "repairs").
- **R1B:** sensitivity of the oracle and global-pool rows to a generic op grid; the random-unit and heuristic-unit
  nulls.
- **R1A:** abstention fit on test_id and evaluated on test_trained/external as the deployable number; state that LLM
  inference is excluded from the cost.
- **General:** keep each round-2 primary to one comparison on one split, written into `round_2/hypotheses.md` before
  any code runs, with n and CI width stated. Several R1 secondary analyses were added after results came in (greedy
  depth-3, cell correlations). They are labelled exploratory, which is correct; keep that discipline.

## Sources consulted for novelty

- [Weight-Space Programming: How LLMs Read and Edit Neural Network Weights (DataCamp)](https://www.datacamp.com/blog/weight-space-programming)
- [Weight-Space Programming: Frontier LLMs Can Natively Read and Rewrite Neural Network Weights](https://themenonlab.blog/blog/weight-space-programming-llms-read-write-neural-network-weights/)
- [Exploring weight space programming: capabilities, generalization and limitations (arXiv 2611.09382 summary)](https://www.opentrain.ai/papers/exploring-weight-space-programming-capabilities-generalization-and-limitations--arxiv-2611.09382/)
- [Weight space programming overhyped debate](https://www.mindstudio.ai/blog/weight-space-programming-overhyped-debate)
- [AI models reading neural network weights benchmark](https://www.mindstudio.ai/blog/ai-models-reading-neural-network-weights-benchmark/)
- [What do LLMs really learn about model weights (LessWrong)](https://lesswrong.com/posts/WnL4nWrS2owzZfwiz/what-do-llms-really-learn-about-model-weights-a)
- [Willison replicates MIT weight-reading test with Qwen 3.8 27B](https://letsdatascience.com/news/willison-replicates-mit-weight-reading-test-on-laptop-with-q-3488f9da)
- [Symmetries in Weight Space Learning: To Retain or Remove? (ICML 2025)](https://icml.cc/virtual/2025/47689)

I found no indexed source for "WSPBench" or for a "Beaker" critique, so they are not cited. The repair and
weight-space learning works above (PRDNN, Arachne, Git Re-Basin, DWSNets, NFN, Tracr meta-models, ROME/MEMIT) are
cited from prior knowledge.
