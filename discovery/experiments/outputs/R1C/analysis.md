## Headline table (solver = prompt-only alignment solver; scale_rl = final trained model, 1 sample/net)
| split | n faulty | solver repair [95% CI] | scale_rl repair [95% CI] | diff (solver-model) [95% paired CI] | solver-only / model-only wins | n clean | solver clean-acc | scale_rl clean-acc | diff clean [CI] |
|---|---|---|---|---|---|---|---|---|---|
| test_id | 422 | 0.931 [0.908,0.955] | 0.829 [0.791,0.865] | +0.102 [+0.062,+0.145] | 60/17 | 78 | 0.936 | 1.000 | -0.064 [-0.128,-0.013] |
| test_larger | 416 | 0.894 [0.863,0.923] | 0.700 [0.651,0.743] | +0.195 [+0.149,+0.243] | 98/17 | 84 | 0.821 | 0.940 | -0.119 [-0.214,-0.024] |
| test_families | 403 | 0.797 [0.757,0.836] | 0.586 [0.538,0.633] | +0.211 [+0.151,+0.270] | 127/42 | 97 | 0.763 | 0.711 | +0.052 [-0.093,+0.196] |
| test_faults | 407 | 0.747 [0.705,0.789] | 0.199 [0.162,0.238] | +0.548 [+0.494,+0.602] | 235/12 | 93 | 0.968 | 0.989 | -0.022 [-0.065,+0.022] |
| test_trained | 403 | 0.069 [0.047,0.094] | 0.667 [0.620,0.712] | -0.598 [-0.648,-0.548] | 2/243 | 97 | 0.010 | 0.948 | -0.938 [-0.990,-0.876] |
| test_external | 415 | 0.087 [0.060,0.113] | 0.513 [0.465,0.561] | -0.427 [-0.480,-0.376] | 8/185 | 85 | 0.012 | 0.718 | -0.706 [-0.800,-0.600] |

## Overall success (faulty + clean, the gym's headline metric)
- test_id: solver 0.932, scale_rl 0.856, noop 0.156, search 0.701; diff +0.076 [+0.042,+0.110]
- test_larger: solver 0.882, scale_rl 0.740, noop 0.168, search 0.705; diff +0.142 [+0.098,+0.184]
- test_families: solver 0.790, scale_rl 0.610, noop 0.194, search 0.743; diff +0.180 [+0.124,+0.234]
- test_faults: solver 0.788, scale_rl 0.346, noop 0.186, search 0.433; diff +0.442 [+0.394,+0.492]
- test_trained: solver 0.058, scale_rl 0.722, noop 0.192, search 0.705; diff -0.664 [-0.706,-0.622]
- test_external: solver 0.074, scale_rl 0.548, noop 0.170, search 0.707; diff -0.474 [-0.522,-0.428]

## Leak check: ep.task.params canonical net vs prompt-parsed canonical net
- test_id: identical per-network success in 100.0% of nets; identical ops 100.0%
- test_larger: identical per-network success in 100.0% of nets; identical ops 100.0%
- test_families: identical per-network success in 100.0% of nets; identical ops 100.0%
- test_faults: identical per-network success in 100.0% of nets; identical ops 100.0%
- test_trained: identical per-network success in 100.0% of nets; identical ops 100.0%
- test_external: identical per-network success in 100.0% of nets; identical ops 100.0%

## Trained / external: what does the solver do?
- test_trained: solver emits [] on 0.2% of nets (faulty 0.0%); noop baseline repair on faulty 0.000 ; solver repair 0.069; scale_rl 0.667
- test_external: solver emits [] on 0.2% of nets (faulty 0.0%); noop baseline repair on faulty 0.000 ; solver repair 0.087; scale_rl 0.513

## test_id: per fault type (n, solver, scale_rl)
- bias_shift: n=70 solver 0.900 scale_rl 0.814
- label_swap: n=14 solver 1.000 scale_rl 0.857
- scale_out_col: n=63 solver 0.905 scale_rl 0.762
- sign_flip_in: n=58 solver 0.914 scale_rl 0.948
- sign_flip_out: n=72 solver 0.972 scale_rl 0.958
- swap_out_cols: n=74 solver 0.946 scale_rl 0.716
- weight_corrupt: n=71 solver 0.930 scale_rl 0.789
## test_id: per depth
- layers=2: n=351 solver 0.960 scale_rl 0.863
- layers=3: n=71 solver 0.789 scale_rl 0.662
## test_id: per family
- argmax: n=51 solver 0.980 scale_rl 0.961
- arith: n=53 solver 0.943 scale_rl 0.774
- compare: n=42 solver 0.857 scale_rl 0.786
- count: n=52 solver 0.962 scale_rl 0.827
- dnf: n=43 solver 0.977 scale_rl 0.930
- equality: n=49 solver 0.857 scale_rl 0.755
- minmax: n=50 solver 0.840 scale_rl 0.740
- parity: n=44 solver 0.977 scale_rl 0.864
- threshold: n=38 solver 1.000 scale_rl 0.842
## test_id: per variant (weights vs weights+failures; solver never sees failing inputs)
- weights: n=224 solver 0.938 scale_rl 0.781
- weights+failures: n=198 solver 0.924 scale_rl 0.884
- solver repair on faulty when 3-layer vs 2-layer shown above

## Pooled compiled splits (id+larger+families+faults), per fault type
                    solver  scale_rl  baseline_search    n
fault                                                     
bias_shift           0.839     0.750            0.233  224
label_swap           1.000     0.920            0.040   25
scale_in_row         0.693     0.297            0.307  192
scale_out_col        0.867     0.694            0.544  196
sign_flip_in         0.898     0.823            1.000  186
sign_flip_out        0.928     0.773            1.000  194
spurious_direction   0.795     0.112            0.308  215
swap_out_cols        0.835     0.530            1.000  200
weight_corrupt       0.875     0.653            0.417  216

## Is scale_rl's per-fault profile correlated with the solver's?
- test_id, 7 fault types: Pearson r(solver, scale_rl) = +0.27, Spearman +0.29; r(search, scale_rl) = +0.29
- compiled splits, 22 (fault x split) cells (n>=8): r(solver, scale_rl) = +0.77, Spearman +0.79; r(search, scale_rl) = -0.01
- compiled splits pooled, 86 (fault x family) cells (n>=8): r(solver, scale_rl) = +0.52, Spearman +0.60; r(search, scale_rl) = +0.31
- (fault x family x depth) cells (n>=8, 88): r(solver, scale_rl) = +0.54
- network-level phi, test_id, scale_rl vs solver: raw +0.176 [+0.045,+0.300], within fault x family x depth strata +0.121 [-0.191,+0.575]
- network-level phi, test_id, scale_rl vs baseline_search: raw +0.065 [-0.030,+0.157], within fault x family x depth strata -0.028 [-0.273,+0.148]
- network-level phi, test_id, scale_rl vs scale_rl_greedy: raw +0.854 [+0.780,+0.916], within fault x family x depth strata +0.858 [+0.800,+0.965]
- network-level phi, compiled pooled, scale_rl vs solver: raw +0.210 [+0.161,+0.258], within fault x family x depth strata +0.091 [+0.040,+0.169]
- network-level phi, compiled pooled, scale_rl vs baseline_search: raw +0.164 [+0.115,+0.214], within fault x family x depth strata +0.045 [-0.024,+0.112]
- network-level phi, compiled pooled, scale_rl vs scale_rl_greedy: raw +0.871 [+0.847,+0.896], within fault x family x depth strata +0.824 [+0.798,+0.870]
- test_id faulty: P(model ok | solver ok) = 0.847 (n=393); P(model ok | solver fails) = 0.586 (n=29)
- union solver OR model = 0.972; model-only fails 0.171; of the model's failures solver fixes 0.833
## test_faults per fault type
- scale_in_row: n=192 solver 0.693 scale_rl 0.297
- spurious_direction: n=215 solver 0.795 scale_rl 0.112
## test_families per family
- interval: n=195 solver 0.600 scale_rl 0.538
- mux: n=208 solver 0.981 scale_rl 0.630