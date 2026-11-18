
### scenario `main`  (n backdoored=340, clean=340)
- white-box verified witness: 100.0% [100.0, 100.0]
-   ... and witness re-verified (independent forward): 100.0% [100.0, 100.0]
-   ... edit hits the true detector neuron: 100.0% [100.0, 100.0]
-   ... removal_ok (0 spec-disagreements on 20k random + neighbourhood + 5k true-trigger samples): 100.0% [100.0, 100.0]
- black-box 1e3 (exact expected): 1.8% [1.5, 2.2]
- black-box 1e4 (exact expected): 13.5% [11.3, 15.6]
- black-box 1e5 (exact expected): 40.9% [36.7, 45.2]
- black-box 1e5 (one Bernoulli draw/net): 40.3% [35.3, 45.6]
- black-box 1e6 (exact expected, context): 66.7% [62.7, 70.8]
- repo weight_rule (knows construction) flags: 100.0% [100.0, 100.0]
- repo weight_rule hits detector: 99.7% [99.1, 100.0]
- CLEAN: verified-witness false alarm: 0/340; unverified top-suspect alarm (influential & p_fire<2^-10): 0/340; repo weight_rule flags: 3/340; nets whose rarest influential neuron is a corner decoy: 0/340
- auditor forward-pass inputs (backdoored): median 321, mean 321 (of which 256 margin-estimation); suspects tested median 1; passes after analysis median 65; max 321
- auditor passes on clean nets (all suspects exhausted): mean 617; audit wall time per net mean 0.003s
- detector rank in auditor's suspect list: rank1 100.0%, rank<=3 100.0%; detector p_fire DP == 2^-k exactly: 100.0%; detector dead-by-exact-analysis never: 0
- backdoored: mean #hidden 13.2; neurons proven dead by exact bound (mean) 6.7; k mean 17.8, n mean 31.5

### scenario `noise3`  (n backdoored=300, clean=300)
- white-box verified witness: 100.0% [100.0, 100.0]
-   ... and witness re-verified (independent forward): 100.0% [100.0, 100.0]
-   ... edit hits the true detector neuron: 100.0% [100.0, 100.0]
-   ... removal_ok (0 spec-disagreements on 20k random + neighbourhood + 5k true-trigger samples): 100.0% [100.0, 100.0]
- black-box 1e3 (exact expected): 1.5% [1.1, 1.8]
- black-box 1e4 (exact expected): 10.5% [8.5, 12.7]
- black-box 1e5 (exact expected): 33.7% [29.5, 37.9]
- black-box 1e5 (one Bernoulli draw/net): 32.3% [27.3, 38.0]
- black-box 1e6 (exact expected, context): 61.9% [57.7, 66.4]
- repo weight_rule (knows construction) flags: 100.0% [100.0, 100.0]
- repo weight_rule hits detector: 100.0% [100.0, 100.0]
- CLEAN: verified-witness false alarm: 0/300; unverified top-suspect alarm (influential & p_fire<2^-10): 0/300; repo weight_rule flags: 2/300; nets whose rarest influential neuron is a corner decoy: 0/300
- auditor forward-pass inputs (backdoored): median 321, mean 321 (of which 256 margin-estimation); suspects tested median 1; passes after analysis median 65; max 321
- auditor passes on clean nets (all suspects exhausted): mean 612; audit wall time per net mean 0.003s
- detector rank in auditor's suspect list: rank1 100.0%, rank<=3 100.0%; detector p_fire DP == 2^-k exactly: 100.0%; detector dead-by-exact-analysis never: 0
- backdoored: mean #hidden 13.4; neurons proven dead by exact bound (mean) 6.9; k mean 18.4, n mean 31.0

### scenario `rescale`  (n backdoored=300, clean=300)
- white-box verified witness: 100.0% [100.0, 100.0]
-   ... and witness re-verified (independent forward): 100.0% [100.0, 100.0]
-   ... edit hits the true detector neuron: 100.0% [100.0, 100.0]
-   ... removal_ok (0 spec-disagreements on 20k random + neighbourhood + 5k true-trigger samples): 100.0% [100.0, 100.0]
- black-box 1e3 (exact expected): 1.6% [1.3, 2.0]
- black-box 1e4 (exact expected): 12.1% [9.9, 14.3]
- black-box 1e5 (exact expected): 36.3% [31.9, 40.7]
- black-box 1e5 (one Bernoulli draw/net): 35.0% [29.7, 40.3]
- black-box 1e6 (exact expected, context): 62.8% [58.3, 67.2]
- repo weight_rule (knows construction) flags: 100.0% [100.0, 100.0]
- repo weight_rule hits detector: 99.7% [98.7, 100.0]
- CLEAN: verified-witness false alarm: 0/300; unverified top-suspect alarm (influential & p_fire<2^-10): 0/300; repo weight_rule flags: 2/300; nets whose rarest influential neuron is a corner decoy: 0/300
- auditor forward-pass inputs (backdoored): median 321, mean 321 (of which 256 margin-estimation); suspects tested median 1; passes after analysis median 65; max 321
- auditor passes on clean nets (all suspects exhausted): mean 625; audit wall time per net mean 0.004s
- detector rank in auditor's suspect list: rank1 100.0%, rank<=3 100.0%; detector p_fire DP == 2^-k exactly: 100.0%; detector dead-by-exact-analysis never: 0
- backdoored: mean #hidden 13.4; neurons proven dead by exact bound (mean) 6.8; k mean 18.1, n mean 32.5

## Stratified by k (main set)
| k bin | n | white-box found | black-box 1e5 exp. | black-box 1e4 exp. | black-box 1e3 exp. | median p_disagree |
|---|---|---|---|---|---|---|
| 12-14 | 95 | 100.0% [100.0, 100.0] | 94.9% | 40.45% | 5.751% | 6.10e-05 |
| 15-17 | 74 | 100.0% [100.0, 100.0] | 52.2% | 8.46% | 0.900% | 7.63e-06 |
| 18-20 | 76 | 100.0% [100.0, 100.0] | 11.8% | 1.28% | 0.130% | 9.54e-07 |
| 21-24 | 95 | 100.0% [100.0, 100.0] | 1.3% | 0.14% | 0.014% | 8.94e-08 |

## Stratified by n_inputs (main set)
| n bin | n nets | white-box found | black-box 1e5 exp. | passes median |
|---|---|---|---|---|
| 24-29 | 137 | 100.0% [100.0, 100.0] | 39.6% | 321 |
| 30-34 | 97 | 100.0% [100.0, 100.0] | 42.8% | 321 |
| 35-40 | 106 | 100.0% [100.0, 100.0] | 40.8% | 321 |

## Fixed-k and fixed-n grids (backdoored only)
| set | n nets | white-box found | black-box 1e5 exp. | black-box 1e6 exp. | median passes |
|---|---|---|---|---|---|
| k12 | 60 | 100.0% [100.0, 100.0] | 99.9% | 100.0% | 321 |
| k16 | 60 | 100.0% [100.0, 100.0] | 53.5% | 99.5% | 321 |
| k20 | 60 | 100.0% [100.0, 100.0] | 5.6% | 42.4% | 321 |
| k24 | 60 | 100.0% [100.0, 100.0] | 0.4% | 4.1% | 321 |
| n24 | 80 | 100.0% [100.0, 100.0] | 37.1% | 68.9% | 321 |
| n32 | 80 | 100.0% [100.0, 100.0] | 38.3% | 63.4% | 321 |
| n40 | 80 | 100.0% [100.0, 100.0] | 45.9% | 65.4% | 321 |

## Family breakdown (main set, backdoored)
- dnf: n=87, white-box found 100.0%, black-box 1e5 exp. 41.9%
- equality: n=81, white-box found 100.0%, black-box 1e5 exp. 37.5%
- parity: n=79, white-box found 100.0%, black-box 1e5 exp. 40.3%
- threshold: n=93, white-box found 100.0%, black-box 1e5 exp. 43.4%

## Paired (main): white-box only 203, black-box(1e5 draw) only 0, both 137, neither 0
white-box misses in main: 0

## Empirical check of exact black-box formula (first 40 main backdoored nets, 1e5 real uniform queries)
- observed detections 17/40 = 42.5% vs exact-formula expectation 44.1%