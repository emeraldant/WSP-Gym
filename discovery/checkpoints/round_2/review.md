# Round 2 review (reviewer)

**What I read:** TASK, README.txt, the round-1 memory and review, the round-2 hypotheses, and reports R2A to R2D.
**What I checked in the scripts:** R2A gen/main, the R2C dev and tune scripts, and the split definitions.
**Read-only checks I ran:** I recomputed R2A's primary from `eval.pkl`, and split it by prompt variant.

## 1. Round-2 primaries

**R2A / H4: rejected. Executed correctly.**
- Seed disjointness is asserted against every seed in `data/`.
- Training uses the test_trained family/fault config.
- Hyperparameters and τ were chosen on val / val_trained only.
- Each answer was scored by `env.score_answer`, with 0 label mismatches.
- I reproduced 40.2% vs 72.2%.

**Fairness gap the report understates.** scale_rl sees failing inputs on 54% of test_trained faulty networks. The ranker never sees them. I re-cut the faulty networks by prompt variant:

| Prompt variant | Faulty nets, n | scale_rl | Ranker |
|---|---|---|---|
| `weights+failures` | 217 | 75.1% | 30.0% |
| `weights` only | 186 | 57.0% | 36.6% |

- On the like-for-like `weights` prompts, the gap in overall success is **−20.1 pts [−27.1, −12.7]** (n = 229).
- H4 is still rejected there, but only narrowly: the CI's upper end is 2.7 pts from the −10 threshold.
- So about a third of the headline −32 is the model's access to failing inputs, not weight reading. Report this as a co-headline.

**Other R2A points:**
- "The model uses information the features don't capture" is speculation; say so.
- R2D's GBM was trained on compiled nets only, yet ranks candidates on trained nets with AUROC 0.82. R2A's compiled-only ranker gets 6.9% top-1 on trained nets. So "collapses across sources" holds for top-1, not for ranking. Qualify it.

**R2B / H5: verdict honest, test weak.**
- Execution is good: fresh disjoint seeds, frozen solver, 1000/1000 bit-identical replay.
- "Supported on point estimates only" is the correct wording. State it plainly: the scale drop clears the 10-pt threshold, the noise drop sits on it (CI −12.8 to −7.4), and pseudo-split is null.
- The pseudo-split arm was a weak test. Default splits are already unequal (α ~ U(0.3, 0.7)), and the solver was built to undo them, so a null was expected.
- Scale and noise arms replicate my round-1 pilot (74 / 82 vs 71 / 82); what is new is the clean-accuracy collapse (97 → 59 / 62 / 20%).
- The transforms outside the gym's symmetry group (pass-through pairs, re-expressed always-on units, an alternative compiler, drift) were not run. So "robust to structure" is shown only for structure the solver was designed for.
- (b) is a properly reported, low-power null, since realised spreads are narrow. #splits at p = 0.044 among 12 uncorrected tests is noise until replicated.

**R2C / H6: supported. Correct, but near-guaranteed.**
- Development touched only val; I confirmed every dev, dbg and diag script hard-codes `val`. The solver file is frozen and its hash checked.
- However, one hand-built weak design will always "lose most of the edge". The 19% is a lower bound for a template-free reader, as the report says.
- The interpretation then overreaches. "≈80% attributable to the compiler" and "the compiled-split gap is entirely the template" are statements about this design, not decompositions.
- The design also uses the generator's lattice and its scale factors. Call it "template-free, generator-aware", not "spec-free".

**R2D / H7: supported. Correct, and largely by construction.**
- Exact hypergeometric expectation, Monte Carlo check, val-only heuristic: well done.
- At B = 25 of ~880 candidates, random search cannot compete with a 65% first guess. Cascade minus model@1 is only +3.6 / +7.2 pts.
- The informative results are three:
  - the full-pool margin of +4.0 [2.2, 6.0];
  - the gap shrinking with B (+8.7 at B = 100, heuristic order, test_external);
  - exhaustive search's held-out P(ok | found) of 0.90 vs 0.97 for the cascade, which is a nice overfitting result.
- "Matched budget" means matched evaluations of the shown network. It excludes LLM inference, and it must say so wherever it is quoted.

## 2. Campaign claims, rounds 0–2

**Solid:**
1. Compiled splits at default obfuscation are solvable without forward passes by inverting the known compiler: 93%, replicated on fresh seeds. A template-free attempt reaches 19%.
2. That shortcut is brittle to rescale and noise, mainly through false repairs on clean networks.
3. On SGD and sklearn networks, no non-LLM weights-only reader tried comes close to scale_rl:

| Reader | Repair on trained nets |
|---|---|
| GBM ranker | 33% |
| R1C solver and spec-free variant | 3–9% |
| scale_rl | 67% |

   Even on weights-only prompts the gap is about 20 pts.
4. Model proposals are a high-precision prior. Forward-pass search beats them only at budgets of a few hundred evaluations.

**Overstated, or needing caveats:**
- "The model is not a template matcher" / "the model is more robust." Its behaviour under hardened obfuscation is untested; the campaign cannot query it.
- Cost savings exclude LLM inference.
- Every R2A gap must carry the failing-inputs asymmetry.
- The R2C attribution figures are design-specific.
- All "non-LLM can't" claims are bounded by the readers actually tried.

## 3. Next experiments (ranked)

1. **Permutation-equivariant learned reader, used as a search prior.**
   - Train a small DWSNet/NFN-style per-unit network on CPU. Tiny nets and ~16k fresh `train_trained` episodes make this feasible.
   - Give it cross-layer path features, which R2A lacked; R2A's label_swap and argmax gaps point to them.
   - Pre-register **verified top-8 from the reader vs scale_rl best-of-8, on weights-only test_trained prompts**. This is the fair test of "the LLM is a uniquely good prior".
2. **Drift continuum plus an alternative compiler.**
   - Adam-fine-tune obfuscated compiled networks for k ∈ {0, 10, 100, 1000} steps, and write a second compiler for 3–4 families.
   - Run the R1C solver, the template-free solver, the GBM ranker and heuristic rich search on the same fresh-seed episodes.
   - Pre-register solver overall success vs k. This turns the compiled/trained dichotomy into a curve, and it is the round-1 item 2b–d that round 2 skipped.
3. **(Out of scope: needs model access; valuable for the authors.)** Get scale_rl rollouts on:
   - the R2B hardened episodes and the drift and alternative-compiler sets;
   - test_trained with failing inputs stripped.

   Also run a matched-compute comparison that counts inference tokens. Only this can establish whether the model is less template-fragile than the solver.

**Small requests:** R2A top-k curves; R2D grid ablation and the B where heuristic search matches best-of-8; R2C per-detector ablation on val.
