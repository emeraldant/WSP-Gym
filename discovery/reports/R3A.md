**Original query**: Test H8 (checkpoints/round_3/hypotheses.md). Scale the compiled audit generator to n_inputs 24-40, k 12-24 with dead and corner decoys. Build a generic white-box auditor that does exact reachability and rare-activation analysis per neuron, constructs a trigger candidate, and confirms it with forward passes against the spec. Compare it with black-box uniform random testing at 1e3, 1e4 and 1e5 queries. PRIMARY: share of backdoored nets with a verified witness, white-box vs black-box at 1e5. Prediction: white-box >= 80%, black-box <= 10%. Also report: cost, false alarms, variation with k and n, and robustness to weight noise x3 and rescale (0.1,10).

**Methodology**
- Scripts: `discovery/experiments/scripts/R3A_gen.py` (generator), `R3A_audit.py` (auditor, black-box, runs), `R3A_analyze.py` (tables, figure). Raw per-network records: `discovery/experiments/outputs/R3A/*.jsonl`; analysis text: `outputs/R3A/analysis.md`. Figure: `discovery/reports/figures/R3A_detection.png`.
- Generator. `audit.make_audit_episode` cannot run at this size because `task.X` would enumerate 2^24 to 2^40 inputs. I reused `AuditConfig`, `_sample_literals`, `_insert`, `_delete`, MARGIN, PUSH_MARGIN, MAX_PUSH, TINY and `nets.sample_task`, `obfuscate`, `pad_net`, and rewrote the episode builder. Changes, all documented in the script docstring:
  1. The task is a spec function on the full space. The family task has 4-7 real inputs and the rest are ignored padding bits (as in `pad_task`, but never materialised).
  2. `obfuscate` runs on the small task and zero padding columns are then added. In audit.py noise also touches the padding columns of ordinary neurons; here they stay exactly 0. Detector, dead and corner rows are dense and noisy exactly as in audit.py.
  3. Whole-domain checks are replaced by exact interval arithmetic. Dead decoy: max pre-activation < 0. Detector and corner: pre > 0 on the entire trigger and pre <= 0 whenever any literal is violated. Clean net exact everywhere: the base network's smallest output margin exceeds the summed worst-case contribution of corner decoys. Backdoor forces the target on the entire trigger. A sample of 50k random and 20k trigger inputs on a handful of nets agreed with these bounds.
  4. Only binary-output families are used (dnf, parity, threshold, equality); `count` is class-valued and was dropped.
  5. 32 tests are sampled from the full space and the episode is redrawn if one fails.
  6. Exact black-box probability p = P_x[net(x) != spec(x)] is computed combinatorially (changed real-bit assignments x 2^free pad bits / 2^n). Detection at q queries is 1-(1-p)^q.
  - Decoys follow audit.py: 5-8 suspects (1 detector + dead, or all dead in a clean net), 0-2 corner decoys, 2-4 obfuscation distractors.
- Auditor (`audit`). It sees only the weights of a one-hidden-layer ReLU net and a callable spec. It does not use the detector construction. Per hidden neuron it computes:
  - exact can-fire (b + sum of positive weights > 0);
  - the exact fraction of the 2^n inputs on which it fires, by DP over weights quantised at 0.01;
  - influence = |outgoing column| x max pre-activation, against the smallest |out-0.5| decision margin on 256 random inputs.
  It then ranks can-fire neurons (influential first, then rarest first). For each suspect, in order (up to 8), it builds x* = argmax pre-activation plus up to 64 neighbours (bits with |w| < max pre are randomised while the neuron keeps firing) and checks forward-pass output against the spec. The first disagreement is the verified witness. The edit zeroes that neuron's outgoing column.
  - Removal is scored by zero disagreements with the spec on 20k random inputs, on the witness neighbourhood (x*, its one-bit flips, 2000 firing fills), and on 5k samples from the true trigger set. The last set is an oracle check.
- Black-box: uniform random queries at 1e3, 1e4, 1e5 (and 1e6 for context). Reported as exact expected detection and as one Bernoulli draw per network. Validation: 1e5 real queries on the first 40 main backdoored nets detected 17/40 = 42.5% against 44.1% expected.
- Data: fresh seeds. main: 340 backdoored + 340 clean, n_inputs uniform 24-40, k uniform 12-24 (exactly as H8). noise3: 300+300 with noise 0.03 (x3). rescale: 300+300 with per-neuron scale (0.1,10). Extra grids (backdoored only): fixed k = 12/16/20/24 with 60 each; fixed n = 24/32/40 with 80 each. CIs are bootstrap 95% over networks (2000 resamples).
- Added secondary baseline: the repo's `weight_rule` (knows the construction), run on the scaled nets.

**Results**

PRIMARY (main, n = 340 backdoored):

| Method | Verified-witness detection |
|---|---|
| White-box generic auditor | 100.0% [100.0, 100.0] (340/340) |
| Black-box 1e5 queries, exact expectation | 40.9% [36.7, 45.2] |
| Black-box 1e5, one draw per net | 40.3% [35.3, 45.6] |
| Black-box 1e4 / 1e3 (exact) | 13.5% [11.3, 15.6] / 1.8% [1.5, 2.2] |
| Black-box 1e6 (context) | 66.7% [62.7, 70.8] |

- Paired at 1e5: white-box only 203, black-box only 0, both 137, neither 0.
- Verdict on the pre-registered prediction: white-box >= 80% holds (100%). Black-box <= 10% FAILS at the stated k range (40.9%). The failure comes from the lower k values: at k 12-14 the trigger fires on about 2^-12 to 2^-14 of inputs, so 1e5 queries expect 6 to 24 hits.

Detection by k (main, backdoored):

| k | n | white-box | black-box 1e5 | 1e4 | 1e3 |
|---|---|---|---|---|---|
| 12-14 | 95 | 100% | 94.9% | 40.5% | 5.8% |
| 15-17 | 74 | 100% | 52.2% | 8.5% | 0.9% |
| 18-20 | 76 | 100% | 11.8% | 1.3% | 0.13% |
| 21-24 | 95 | 100% | 1.3% | 0.14% | 0.01% |

Fixed-k sets (60 each): black-box 1e5 = 99.9% (k=12), 53.5% (16), 5.6% (20), 0.4% (24). White-box 100% each. The black-box "<=10%" prediction would hold only for k >= about 20, and 1e6 queries lift k=20 to 42%.

By n_inputs: n 24-29 / 30-34 / 35-40 gives black-box 39.6 / 42.8 / 40.8% and white-box 100% each. Fixed n = 24/32/40 gives black-box 37.1 / 38.3 / 45.9%. Black-box rate does not depend on n, since the trigger rate is 2^-k times the changed share. Rarity is set by k, not n. White-box has no n or k dependence in these runs.

Cost and false alarms:
- Auditor forward passes. Median and mean 321 input vectors on backdoored nets: 256 for the margin estimate plus 1 + 64 for the first suspect tested. In 340/340 the detector was rank 1, so 1 suspect was tested. Analysis wall time about 3 ms per net (DP is exact and cheap). On clean nets the auditor exhausts all suspects, mean about 617 passes (about 4-7 non-dead neurons tested, no witness).
- Matched budget: at 321 black-box queries, expected detection is about 0.6% in main (1.8% at 1e3).
- False alarm on clean nets, main: 0/340 verified witnesses (as expected: clean nets are exact by construction). Unverified top-suspect alarm (influential and p_fire < 2^-10): 0/340. The repo's `weight_rule` flags 3/340 clean nets, and hits the detector in 99.7% of backdoored.
- Exact analysis: on average 6.7 of 13.2 hidden neurons are proven unable to fire (the dead decoys). The detector's DP fire fraction equals 2^-k exactly in 100% of nets, confirming the DP and the generator's exact firing set. The rarest influential can-fire neuron was the detector in 100%, and no corner decoy outranked it (corner decoys are rare but their column x max-pre is below the decision margin).
- Removal: edit hits the true detector in 340/340; removal_ok (all three checks clean) 340/340.

Robustness (same generator, regenerated at the perturbed settings):

| Scenario | white-box | removal_ok | black-box 1e5 | clean false alarm (verified) |
|---|---|---|---|---|
| main | 100% (340) | 100% | 40.9% | 0/340 |
| noise x3 (0.03) | 100.0% [100,100] (300) | 100% | 33.7% [29.5, 37.9] | 0/300 |
| rescale (0.1, 10) | 100.0% [100,100] (300) | 100% | 36.3% [31.9, 40.7] | 0/300 |

Influence uses |col| x max-pre, which is invariant to rescaling, and the DP uses the actual weight grid, so both variants leave detector rank 1 and an exact 2^-k fire fraction. No degradation was observed at all.

**Interpretation**
- H8 is half confirmed. A generic reachability-plus-influence auditor finds a verified witness and the correct removal on 100% of backdoored nets at n 24-40, k 12-24, with about 65 forward passes beyond a 256-sample margin estimate. The black-box "<=10%" part was mis-calibrated: uniform testing at 1e5 queries still catches about 41% in the k 12-24 range. White-box wins a lot only for k >= 18 (black-box 12% -> 1%). At k = 12-14 black-box is nearly as good (95%). So the honest statement is "reading weights beats testing by a margin that grows as 2^k", not "testing cannot find these".
- The 100% is not evidence of a hard-won capability. It shows that this compiled decoy design is fully defeated by exact reachability: dead decoys are provably unfireable (a single bound), corner decoys fire but cannot change the output, and the detector is the one neuron that is both rare and influential. The pre-registered "weights beat testing" result is therefore a statement about the generator's decoy construction, which is the same thing R1C/R2C showed for repair (the compiler is the weak point). The generic auditor did NOT need a template, but it exploits that hidden neurons are first-layer, binary inputs and single-hidden-layer, so exact DP applies. It would not transfer to SGD-poisoned (distributed) backdoors, which is H9's question; I did not test that.
- The auditor is "generic" about the detector but not about the network class: it assumes one hidden layer of ReLUs with a linear output and binary inputs. The influence test also uses a sample-based margin (256 random inputs), which on larger or harder specs may misjudge decision margins; ranking would then fall back to verification cost (more suspects tested). I did not ablate DP rarity ranking against random order among influential neurons, so I cannot say how much of the 1-suspect cost depends on the rarity DP.
- Caveats: (a) generator changed as listed (no noise on padding columns of ordinary neurons; binary families only; exactness proven by bounds, not enumeration, with a few hundred thousand sampled inputs as spot checks). (b) The no-false-alarm claim is by construction, since a verified witness is a real disagreement; 0/340 is a property of the pipeline, not an empirical discovery. The clean nets are exact, which is guaranteed by the generator's bound. (c) Black-box 1e5 numbers are exact expectations over the generator's trigger sizes (k uniform 12-24); the ">=10%" outcome is sensitive to that range. (d) The perturbed scenarios regenerate nets with different noise/scale, rather than perturbing the same nets; the reject rate of the generator in those scenarios was not logged.
- Follow-ups: ablate the auditor (no DP / no influence test) to see which analysis does the work; make decoys that defeat exact analysis (e.g. a live detector that needs the other first-layer neurons' states, or a two-layer detector) so white-box needs search; compare with a frontier-LLM auditor reading the same weights as text, which is the setting the literature cites; and run H9's distributed backdoor where no single neuron exists.
