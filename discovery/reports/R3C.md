**Original query**: Test H10 (round_3/hypotheses.md). Train a neuron-level fault localiser only on benign repair faults (compiled and trained source, no backdoors), weight-only scale-free features adapted from R2A, apply zero-shot to data/test_audit.jsonl and data/test_audit_sgd.jsonl. PRIMARY: top-1 localisation of the detector on backdoored compiled audit nets vs chance (1/#hidden) and vs audit_baselines stat_outlier, plus detection AUROC (max suspicion, backdoored vs clean). SGD audit: localisation = zeroing the top-ranked neuron's outgoing column gives recovery >= 0.95 on the full domain. Upper bound: same model trained on val_audit / val_audit_sgd. Also weight_rule and search baselines, and feature importances.

**Methodology**
- Scripts: discovery/experiments/scripts/R3C_{feat,gen,valsgd,main,ub}.py; outputs in discovery/experiments/outputs/R3C/ (main.log, ub.log, baselines.txt, results_primary.json, results_ub.json, importance.json). The R2A pickles hold candidate-edit rows without neuron labels, so they were not reused; I regenerated a neuron-level set (reusing R2A_feat.NetInfo neuron features).
- Training data (benign only): GymConfig repair episodes, default train families/faults, size alternating 0/1, seeds 82,000,000+ (trained) and 83,000,000+ (compiled), disjoint from all splits. 3000 episodes per source, clean episodes and label_swap (output units only) dropped: 4795 faulty nets, 42,304 hidden neurons, 5,650 positive. Label = every hidden neuron (layer, idx) the fault touches (sign_flip/scale/bias_shift: the neuron; swap_out_cols/spurious_direction: all 2-3 neurons; weight_corrupt: the row neuron if layer<L-1 and the column neuron if layer>=1).
- Features (PRIMARY, 46 columns): the 23 R2A per-neuron features (row/col norms and ranks, rescale-invariant gain z, bias/row-norm, sign consistency, matched-twin cosines, cluster consistency, interval-arithmetic pre-activation bounds, dead/always-on, path share ...) plus a within-net robust-z version of each (width-free). Secondary "+extra" adds 13 generic fan-in/peak-margin/effect features (these are close to the weight_rule construction, so they were NOT the primary). Also "R2A raw (23)" and per-source models.
- Model: sklearn HistGradientBoosting (200 iters, lr .06, 15 leaves) on neurons; per net, suspicion = predicted probability; top-1 = argmax; detection score = max over neurons.
- Evaluation: test_audit (500 nets, 247 backdoored; mean 13.5 hidden neurons, chance 1/n = 7.4%), test_audit_sgd (500, 241 backdoored). Localisation hit for compiled = argmax equals fault_info.neuron (identical to the zero-column recovery criterion up to 1 net); for SGD = score_answer recovery >= 0.95 of zero_col(top1) (episodes only guarantee that SOME single-op edit works; zeroing a column works for some neuron in 81.7% of backdoored SGD nets, so that is the ceiling). Bootstrap 95% CIs over networks (1000 resamples; AUROC 500, resampling both classes). stat_outlier ranking = per-neuron max robust |z| over its four features (its top-1 reproduces audit_baselines' stat_always removal exactly: 0.049 / 0.432). Upper bound: same model trained on val_audit (300 nets) or a val_audit_sgd I generated (300 nets, seeds 13,000,000-13,000,299 = train_audit_sgd stream, disjoint from test; 165 backdoored), all neurons of clean nets as negatives; for SGD labels = neurons whose column zeroing recovers.

**Results**

Repair-domain sanity (held-out 20% nets, primary feats): top-1 neuron hit 58.5% (compiled, n=479) and 57.8% (trained, n=481) vs chance 15-16%. So the localiser works in-distribution.

PRIMARY, test_audit (compiled), n_bd=247 / n_clean=253:

| scorer | top-1 = detector (95% CI) | AUROC (95% CI) |
|---|---|---|
| chance 1/n | 7.4% | 0.50 |
| **benign-only localiser (primary)** | **2.4% (0.8-4.5)** | **0.43 (0.38-0.48)** |
| stat_outlier (ranking) | 4.5% (2.0-7.3) | 0.50 (0.45-0.55) |
| paired diff primary minus stat_outlier | -2.0 pts (-5.3, +1.2) | |
| +extra feats / R2A raw / compiled-only / trained-only | 0.4% / 5.7% / 0.8% / 1.6% | 0.42 / 0.50 / 0.44 / 0.41 |
| UB: trained on val_audit (primary feats) | 100% (100-100) | 0.996 (0.99-1.0) |
| repair + val_audit backdoors | 98.8% | 0.991 |
| weight_rule (hand-written, knows the construction) | 99.6% | 1.00 |

H10 PRIMARY is falsified: zero-shot top-1 is at or BELOW chance (2.4% < 7.4%), detection AUROC is 0.43 (at/below 0.5). Where the top-1 lands on backdoored nets: 204 ordinary (function-carrying/distractor) neurons, 33 corner decoys, 4 dead decoys, 6 detector. Base rates are 6870 neurons, 3010 dead decoys, 503 corner. The localiser systematically prefers neurons that carry real function (path_share) and avoids the detector. Single features give the same picture (top-1: path_share 19.8%, rn 13.4%, gain_z 16.6%, iv_hi 0.4%; AUROC all 0.48-0.54).

PRIMARY-analogue on test_audit_sgd (recovery criterion), n_bd=241 / n_clean=259; chance = mean fraction of neurons whose zeroing works = 6.4%; ceiling (some column works) 81.7%:

| scorer | top-1 zero-column recovery (95% CI) | top-3 | AUROC (95% CI) |
|---|---|---|---|
| **benign-only localiser (primary)** | **45.6% (39.4-51.9)** | 62.2% | 0.51 (0.46-0.56) |
| stat_outlier ranking | 43.2% (36.9-49.8) | 56.8% | 0.39 (0.34-0.44) |
| paired primary minus stat_outlier | +2.5 pts (-3.7, +8.3), n.s. | | |
| random | 4.6% | 18.3% | 0.48 |
| +extra / R2A raw 23 / compiled-source-only | 26.6% / 29.5% / 0.4% | | 0.43 / 0.34 / 0.48 |
| trained-source-only (secondary, post hoc) | 63.5% (57.3-69.3) | 72.6% | 0.69 (0.64-0.74) |
| UB: trained on val_audit_sgd | 75.9% (70.5-81.3) | | 0.958 (0.94-0.975) |
| repair + val_audit_sgd backdoors | 70.1% | | 0.78 |
| weight_rule | 12.0% | 24.5% | 0.54 |

So for SGD-distributed backdoors localisation transfers far above chance (45.6% vs 6.4%) but it is no better than the magnitude-outlier baseline (paired diff +2.5, CI spans 0). Detection AUROC does not transfer (0.51). Using only trained-source repair faults is better (63.5% top-1, AUROC 0.69) and compiled-source repair faults actively hurt (0.4%): the compiled-source repair prior is anti-correlated with SGD backdoors. Single features: path_share 49.8%, cn_rel 48.1%, gain_z 36.5% top-1 (so one magnitude feature alone matches the learned model); iv_hi_rel has AUROC 0.63 for detection only.

Baselines for context (audit_baselines.run_all, finetune excluded; baselines.txt). Detection / false alarm / removal:
- test_audit: stat_outlier 0.82/0.83/0.04; stat_always removal 0.049; stat_informed 1.0/1.0/0.154; search_tests (32 passing tests) 0/0/0 (never sees a failure); search_oracle (full domain) 1.0/0/1.0; weight_rule 1.0/0/0.996; random_tests 32/256/1024 queries detect 22.7/76.5/95.5% (FA 0).
- test_audit_sgd: stat_outlier 0.82/0.84/0.38; stat_always removal 0.43; stat_informed 1.0/0.47/0.456; search_tests 0/0/0; search_oracle 0.975/0/0.959; weight_rule 0.075/0/0.075 (does not transfer to SGD); random_tests 25.7/73.0/97.9%.

Feature importance (permutation, AUROC drop, primary model):
- Dominant on all three: path_share (repair 0.105, test_audit 0.131, test_audit_sgd 0.185). Second: iv_hi_rel (z-scored on SGD): 0.099 on compiled, 0.066 on SGD. Then match_cos_row (twin-neuron cosine), rn_rel, cn_rel, bias_rel, row_nnz, match_col_ratio at <=0.02. The repair-importance numbers are in-sample (computed on the last 300 training nets) so are only indicative.
- What transfers to SGD backdoors: path-share / column-norm style "this neuron matters a lot to the output" (alone 48-50% top-1) and upper interval bound. What does not transfer to compiled: the same features point away from the detector (plausible explanation, not tested: the compiled design equalises row/column magnitudes of detector, dead and corner decoys, and top-1 mostly lands on ordinary function neurons). Only the arithmetic "can it ever fire" check (weight_rule / UB training on backdoors) finds it.

**Interpretation**
- Pre-registered H10 does not hold for the compiled audit: benign repair faults give no signal about the planted detector (below chance, AUROC 0.43), even though the same localiser reaches 58% top-1 on its own task and a localiser trained on 300 val backdoors reaches 100%/AUROC 0.996. Repair skill (finding the neuron whose weights are wrong) does not transfer to finding a neuron that is correct-looking and dormant. The hand-written weight_rule is the only zero-shot method that works on compiled nets, and it is construction-specific (7.5% on SGD).
- On SGD-poisoned nets localisation partially transfers (45.6% vs 6.4% chance) but this is explained by "large-effect neuron" magnitude statistics that stat_outlier already uses; AUROC for detection stays near 0.5, so a benign-trained localiser cannot decide whether a network is poisoned. The generated in-distribution upper bound (75.9% top-1, AUROC 0.96) shows the signal exists in the weights; the gap is a distribution-shift gap in training signal, not information. The source mismatch matters: trained-source repair data alone gives 63.5%/0.69, compiled-source data 0.4%.
- Caveats: (1) a few features (path_share, iv_hi_rel) dominate; no hyperparameter tuning or feature-set search beyond the 4 variants listed, and the variants beyond "primary" were chosen post hoc, so treat their numbers as exploratory. (2) For SGD, success criterion is single-column zeroing; backdoors needing other edits count as misses (ceiling 81.7%). (3) val_audit_sgd was generated by me (n=300, 165 backdoored) and is not a shipped split; test nets share the generator with it. (4) Single training run/seed for each model; no CI on model-seed variance. (5) UB near 100% on compiled partly reflects that the detector is a deterministic construction. (6) Importance is permutation on the same sets used for evaluation (descriptive only). (7) Training nets include 2-hidden-layer nets; audit nets are one hidden layer padded to 9-10 binary inputs, so features like dom_w interval scaling are shifted in distribution.
- Follow-ups: train the localiser on repair faults PLUS a "dormant/benign-looking" pathology class (as in the Weight Oracles protocol, which trains on pathologies close to the target) and check whether compiled transfer appears; test combining the interval "can it fire" feature as a hard-coded feature with trained-source repair data for SGD; evaluate detection via a verified witness (H8/H9) rather than suspicion score.
