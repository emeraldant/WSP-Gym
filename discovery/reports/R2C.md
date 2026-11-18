**Original query**: Test H6 (does the solver need the compiler?). Build a spec-free variant of the R1C solver that never calls the spec parser or `build_task`. It may use the shown weights, the architecture and the input range, but no template. Replace template matching with structure inferable from the weights alone (cluster/split/cancel detection, sign and pair consistency, robust outlier detection, dead-neuron interval bounds, a learned-free decision rule for `[]` vs edit). Tune thresholds on `data/val.jsonl` only, freeze, run the test splits once. PRIMARY: test_id faulty repair, spec-based (`outputs/R1C/solver_test_id.csv`) vs spec-free, paired bootstrap. Also report overall success including clean controls, all six test splits each against `runs/final/scale_rl`, and test_trained / test_external specifically. Interpret: how much of the 93% is "having the compiler"?

**Methodology**

*Pre-registration (round_2/hypotheses.md, H6).* Hypothesis: the spec-free solver loses most of the edge. PRIMARY: test_id repair rate on faulty nets, spec-based R1C solver minus spec-free solver, paired over the 422 faulty networks, 5,000-resample bootstrap (seed 0). Everything else below is labelled secondary/exploratory.

*What the spec-free solver sees.* Only `e.net` (the shown weights and biases), the layer shapes, and the input range (`hi` = 1 if the input description says "binary", else 9; this is the same quantity the R1C solver was given). It never reads `task.spec`, `family`, `params`, `build_task`, `ep.clean`, `inverse_edit`, fault fields or failing inputs, and it never evaluates any network on inputs. The only semantic computation is interval (box) range analysis of hidden pre-activations. The frozen file has no import other than numpy (`outputs/R2C/R2C_solver_FROZEN.py`, md5 3855718b...).

*Solver design (`scripts/R2C_solver.py`, about 330 lines).* It replaces the template with an integer-lattice prior plus structure found in the weights.
1. Per-neuron unit: each hidden neuron's row is rescaled by the unit `u` that makes its entries nearest to small integers (robust cost with a cap, plus a complexity penalty). This undoes the per-neuron rescale. Output rows are not rescaled.
2. Clustering and collapse: alive neurons whose normalised rows agree (max abs difference < 0.3) are clustered as split copies or cancel pairs; their outgoing columns are summed in canonical coordinates, and cancel groups (sum about 0) are dropped. Dead neurons are found by interval upper bound (UB <= 0.1).
3. Anomaly detectors, each emitting a candidate edit with a strength:
   - off-lattice column entries in canonical coordinates (col_entry) -> scale the column, or set one entry, snapped to the lattice;
   - off-lattice bias or row entry (row_bias, row_entry) -> set the bias with a generic small-integer prior (0, -1, -2, 1, ...), or one entry;
   - swapped-columns hypothesis: a pair of neurons whose columns are off-lattice but become lattice when exchanged (swap);
   - cluster consistency: equal-magnitude same-sign column pair (cl_equal, flip), non-stair mixed-sign non-cancelling cluster (cl_mixed, rescale a member so the pair cancels);
   - dead-neuron test: a dead neuron with a non-zero column, all-non-positive row, whose sign-reversed row is alive, flipped (dead_flip).
4. Decision rule (learned-free): emit the single strongest anomaly if its strength >= EMIT = 0.2, else `[]`. Detectors that were tried and dropped on val because they were net-negative: odd-sign-out column check (precision 1/10), cluster non-parallel columns (an identical-row cluster only constrains the column sum, so this is not an anomaly), near-miss dead neurons, ambiguous dead flips (3/39, 19 of which were clean controls).
5. Not covered at all: label_swap (no way to know the canonical output order without a template), sign flips that stay alive, lattice-preserving corruptions (zero, negate, copy).

*Tuning discipline.* Thresholds were set only by looking at the 300 val episodes (detector precision tables, then a 192-point grid over EMIT, W_COL, W_ROW, W_SW, W_CL, HALF_W; all configurations scored 0.33-0.36 overall on val, so the choice is not sensitive). The solver was frozen (file hash asserted in `R2C_final.py`), then run once on the six test splits. No test split was run during development. Val, 300 episodes (236 faulty): faulty repair 0.212, clean accuracy 0.891, overall 0.357.

*Honest caveat on "spec-free".* The design uses my knowledge of the gym's symmetry group (what splits, cancel pairs, dead neurons and rescales look like) and the generator's integer-lattice weights. Two small constants echo generator knowledge: the scale-factor grid {1/3, 0.4, 2.5, 3, 0.5, 2, 1.5, ...} used when fitting a column scale, and the bias prior ordering. So this is "no spec, no `build_task`", not "no knowledge of the gym". It is one hand-built attempt, so its numbers are a lower bound for what a spec-free reader could do.

*Statistics.* Paired bootstrap over networks, 5,000 resamples, seed 0. "Repair" = success on faulty nets. "Clean accuracy" = answered `[]` on clean controls. "Overall" = success over faulty + clean (n = 500 per split). scale_rl = `runs/final/scale_rl/<split>.jsonl`, sample 0, joined on seed. Spec-based = `prompt_success` in `outputs/R1C/solver_<split>.csv`.

**Results**

PRIMARY (test_id, n = 422 faulty, paired):

| | repair [95% CI] |
|---|---|
| spec-based R1C | 0.931 [0.908, 0.955] |
| spec-free R2C | **0.192 [0.154, 0.230]** |
| spec-based minus spec-free | **+0.739 [+0.697, +0.780]** |
| spec-free minus scale_rl (0.829 [0.794, 0.865]) | -0.637 [-0.685, -0.588] |

Discordant pairs: 316 nets repaired by spec-based only, 4 by spec-free only. H6 (the spec-free variant loses most of the edge) is supported, and then some: it does not merely lose the +10 pt margin over scale_rl, it loses 74 points of absolute repair. Pre-registered prediction met.

All splits (faulty repair / clean accuracy / overall success incl. clean; 95% CI on repair and overall):

| split | n faulty / clean | spec-based repair | spec-free repair | scale_rl repair | spec-based clean | spec-free clean | scale_rl clean | spec-based overall | spec-free overall [CI] | scale_rl overall [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| test_id | 422/78 | 0.931 | 0.192 [0.154,0.230] | 0.829 | 0.936 | 0.859 | 1.000 | 0.932 | 0.296 [0.256,0.336] | 0.856 [0.824,0.886] |
| test_larger | 416/84 | 0.894 | 0.142 [0.108,0.176] | 0.700 | 0.821 | 0.500 | 0.940 | 0.882 | 0.202 [0.166,0.238] | 0.740 [0.702,0.778] |
| test_families | 403/97 | 0.797 | 0.216 [0.176,0.256] | 0.586 | 0.763 | 0.804 | 0.711 | 0.790 | 0.330 [0.292,0.372] | 0.610 [0.568,0.652] |
| test_faults | 407/93 | 0.747 | 0.103 [0.074,0.135] | 0.199 | 0.968 | 0.806 | 0.989 | 0.788 | 0.234 [0.198,0.274] | 0.346 [0.306,0.388] |
| test_trained | 403/97 | 0.069 | 0.032 [0.017,0.050] | 0.667 | 0.010 | 0.031 | 0.948 | 0.058 | 0.032 [0.018,0.048] | 0.722 [0.684,0.762] |
| test_external | 415/85 | 0.087 | 0.048 [0.029,0.070] | 0.513 | 0.012 | 0.176 | 0.718 | 0.074 | 0.070 [0.048,0.094] | 0.548 [0.502,0.592] |

Paired differences, spec-free minus scale_rl (repair on faulty / overall incl. clean):

| split | repair diff [95% CI] | overall diff [95% CI] |
|---|---|---|
| test_id | -0.637 [-0.685,-0.588] | -0.560 [-0.606,-0.512] |
| test_larger | -0.558 [-0.611,-0.507] | -0.538 [-0.584,-0.488] |
| test_families | -0.370 [-0.432,-0.305] | -0.280 [-0.338,-0.220] |
| test_faults | -0.096 [-0.138,-0.057] | -0.112 [-0.150,-0.076] |
| test_trained | -0.635 [-0.682,-0.586] | -0.690 [-0.730,-0.648] |
| test_external | -0.465 [-0.516,-0.414] | -0.478 [-0.524,-0.430] |

Spec-free is significantly below scale_rl on every one of the six splits, on both repair and overall success. Spec-based minus spec-free (paired, repair): +0.739, +0.752, +0.581, +0.644 on the four compiled splits; +0.037 [+0.010,+0.067] on test_trained and +0.039 [+0.007,+0.070] on test_external, i.e. even on trained/external nets the spec-based solver is slightly ahead, not behind.

The question about test_trained / test_external, where a spec-free reader might beat the template solver: it does not. Spec-free repair is 3.2% / 4.8%, spec-based 6.9% / 8.7%, scale_rl 66.7% / 51.3%. Both solvers are essentially a no-op quality reader on SGD and sklearn nets. The spec-free solver emits an edit on nearly every trained net (clean accuracy 3.1%), so its overall success (0.032 / 0.070) is below the always-`[]` baseline (the clean share, 0.194 / 0.170). The lattice prior has nothing to hold on to when weights are not integer-structured.

Compiled-split overall success vs the always-`[]` baseline: test_id 0.296 vs 0.156, test_larger 0.202 vs 0.168, test_families 0.330 vs 0.194, test_faults 0.234 vs 0.186. The spec-free solver is above no-op only on test_id and test_families by a clear margin. On test_larger its clean accuracy falls to 0.50 (more distractors and wider nets make the lattice detectors fire on clean controls).

test_id by fault type (faulty repair; spec-based / spec-free / scale_rl; n):

| fault | spec-based | spec-free | scale_rl | n |
|---|---|---|---|---|
| bias_shift | 0.900 | 0.229 | 0.814 | 70 |
| label_swap | 1.000 | 0.000 | 0.857 | 14 |
| scale_out_col | 0.905 | 0.381 | 0.762 | 63 |
| sign_flip_in | 0.914 | 0.241 | 0.948 | 58 |
| sign_flip_out | 0.972 | 0.111 | 0.958 | 72 |
| swap_out_cols | 0.946 | 0.176 | 0.716 | 74 |
| weight_corrupt | 0.930 | 0.085 | 0.789 | 71 |

test_id by depth: 2-layer 0.960 / 0.214 / 0.863 (n = 351); 3-layer 0.789 / 0.085 / 0.662 (n = 71). By family, spec-free ranges 0.12 (dnf) to 0.24 (minmax); spec-based 0.84-1.00.

Where spec-free works: faults that leave a detectable lattice or cancellation violation. scale_out_col (38%; a factor of 2.5, 0.4 or 0.3 breaks the integer lattice, a factor of 3 does not), sign_flip_in on dead-after-flip neurons, swapped columns with distinct column scales, and equal-magnitude cancel-pair flips. Where it cannot work: sign_flip_out, weight_corrupt and label_swap, which keep every weight on the lattice and need a template to know what the value should be. Detector precision on test_id (successes among emissions; a "clean" emission is a false alarm): cl_mixed 21/57 (0 clean), cl_equal 9/21 (1 clean), dead_flip 15/34 (2 clean), row_bias 14/39 (0 clean), swap 13/49 (5 clean), col_entry 67/273 (61 clean), row_entry 9/26 (9 clean).

Secondary: union and complementarity. spec-free OR scale_rl repairs 0.851 on test_id (scale_rl alone 0.829). Spec-free adds 2 points over the model, which is about as much as noise.

Secondary: stability. Val 0.212 repair / 0.891 clean / 0.357 overall; test_id 0.192 / 0.859 / 0.296. A 192-point threshold grid on val moved overall success between 0.33 and 0.36, so the result is not a threshold artefact.

Files. Per-network outputs: `discovery/experiments/outputs/R2C/specfree_<split>.csv`; frozen solver `outputs/R2C/R2C_solver_FROZEN.py`; summaries `outputs/R2C/summary.csv`, `test_id_by_fault.csv`, `test_id_by_family.csv`, `test_id_merged.csv`. Figure: `discovery/reports/figures/R2C_main.png`. Scripts: `discovery/experiments/scripts/R2C_{solver,run,final,analyze,tune,diag,diag2,diag3,dbg*,dev1}.py`.

**Interpretation**

- **How much of the 93% is "having the compiler"?** Nearly all of it. On test_id, going from no template to the exact template takes the same alignment machinery (cluster, unit, residual) from 19% to 93% repair. Counting from the no-op floor of 0%, about 80% of the template solver's repair is attributable to knowing the compiled form, and 20% to structure that is visible in the weights alone (integer lattice, cancel pairs, dead-neuron bounds, swapped columns). Overall success tells the same story (0.296 vs 0.932, no-op 0.156). The spec-free solver is far below scale_rl (-64 pts repair, CI excludes 0 on all six splits), so the compiled-split gap in R1C (+10 pts for the template solver) is entirely the template, not the reading skill.
- **What a template removes.** It resolves the ambiguities that weights cannot: which of several lattice-equivalent values is right (sign flips, zeroed/negated/copied entries, bias shifts), which output is which (label swaps), and unit choice. Spec-free repairs concentrate on faults that break the weight lattice (scale/bias off-lattice, swapped columns with different scales) and fail on the lattice-preserving ones (sign_flip_out 11%, weight_corrupt 9%, label_swap 0%). Those are exactly the faults where scale_rl is strong (96%, 79%, 86%). The model is therefore not reading weights the way a lattice reader would.
- **Trained and external nets.** The prediction that a spec-free reader might help there is not supported. The lattice prior is a compiled-net artefact; on SGD and sklearn nets it fires on everything (clean accuracy 3% / 18%) and repairs 3-5%. What scale_rl does on test_trained (67%) / test_external (51%) is not obtainable from weight structure of this kind. A learned reader (R1 review item 1) is the remaining untested non-LLM baseline.
- **Reading for the campaign.** R1C established that compiled splits are solvable by inverting a known generator. R2C adds that this is a statement about the generator's uniqueness, not about weights being readable by structure alone. The template is the load-bearing component. Together they say: compiled-split scores measure how much of one compiler a reader knows; test_trained / test_external remain the splits that measure reading beyond a template.
- **Caveats.** (1) One hand-engineered spec-free design, a better spec-free reader (e.g. learned from the 3-5k `train` episodes, or a gadget library inferred by clustering canonical rows across many nets) could do more, so 19% is a lower bound on what is possible without the spec, not a ceiling. (2) It still encodes the symmetry group and parts of the generator's lattice, so "spec-free" is not "gym-free". (3) The decision rule is crude: clean accuracy is 0.86 on test_id but 0.50 on test_larger and 0.03 on trained nets; a lattice-ness gate would raise overall success on trained nets to the no-op level but not repair rate. (4) scale_rl is one sample in direct mode; it sees failing inputs on about half the nets and the solvers never do. (5) The reported spec-based numbers are the R1C ones (not re-run); the pairing is on seed. (6) Several constants in the column-scale fit echo generator factors (flagged above). (7) Occasional divide-by-zero warnings from a harmless `lam[0]/lam[1]` ratio; no episode errored (err = 0 on all splits).
- **Suggested follow-ups.** (a) Learned spec-free reader trained on `train` and `train_trained` streams as a stronger test of "no compiler" (this is where the remaining headroom is). (b) Stress the R1C solver with alternative compilers (reviewer item 2c) to see if the 93% is specific to one compiler. (c) Report a spec-free no-op-gated variant using a lattice-ness score to recover the clean accuracy on trained nets. (d) Per-fault ablations of each detector (add them back one by one on val) to apportion the 19%.
