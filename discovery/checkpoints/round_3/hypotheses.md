# Round 3 hypotheses — toward auditing (pre-registered; written before any round-3 code runs)

Motivation (user thesis): future alignment may rest on swarms of weight-auditing agents, so the question is what
reading weights buys an auditor over testing behaviour. Recent literature (checkpoints/round_3/literature.md):
frontier LLMs detect planted poison from weights but never find the trigger (arXiv 2611.21449); an oracle trained on
benign pathologies detects unseen backdoors zero-shot, AUROC 0.81-0.93 (arXiv 2610.07334). WSP-Gym's audit splits
(9-10 inputs, triggers firing on 1/32-1/256 of inputs) are small enough that black-box testing finds every trigger,
and no model has been evaluated on them.

## H8 — Reading weights finds rare triggers that testing cannot (R3A)
Scale the compiled audit generator to large input spaces with rare triggers (n_inputs about 24-40; k = 12-24 literals,
so triggers fire on 2^-12 to 2^-24 of inputs; keep the decoy design: dead decoys and corner decoys). A GENERIC
white-box auditor (no spec template, no knowledge of the detector construction beyond "a ReLU network") does:
reachability/rare-activation analysis per neuron (can it fire? on what fraction of inputs — exact DP over binary
inputs for first-layer neurons? does firing flip the output?), then constructs a candidate trigger input and confirms it
with ONE forward pass against the spec. Output: a verified witness (an input where the network disagrees with the
spec) plus an edit removing the backdoor.
PRIMARY: on backdoored test networks, share with a verified witness found by the white-box auditor vs share detected by
black-box uniform random testing at 10^5 queries; false-alarm rate on clean networks is 0 by construction for any
verified witness, and is reported anyway. Prediction: white-box >= 80%, black-box <= 10%.

## H9 — Distributed (SGD-poisoned) backdoors: white-box witness search vs testing, and swarms of verified auditors (R3B)
Poison SGD-trained networks at the largest scale where training is feasible on CPU (sampled training data; n_inputs
about 14-20, trigger rate 2^-8 to 2^-14). There is no single detector neuron. Auditors: black-box random testing;
generic white-box witness search guided by the weights (e.g. per-neuron rare-activation analysis plus greedy or beam
search over the hypercube maximising "network disagrees with spec", using forward passes, budget-matched to the
black-box tester); and weight-statistics flags. Each auditor must emit a verified witness to count.
PRIMARY: detection with a verified witness at matched forward-pass budget (10^3 and 10^4), white-box vs black-box,
paired over networks. SECONDARY (swarm): union of K diverse verified-witness auditors vs the best single one at the
same total budget. A union of verified claims has no added false-alarm cost, which is what makes swarms composable.

## H10 — Repair skill transfers to auditing (Weight Oracles protocol, non-LLM) (R3C)
Train a fault localiser ONLY on benign repair faults (WSP-Gym repair episodes, no backdoors), with weight-only features
(reuse/adapt the R2A feature set and ranker), then apply it zero-shot to data/test_audit.jsonl and data/test_audit_sgd.jsonl.
PRIMARY: top-1 localisation of the backdoor on backdoored test_audit networks (the hidden neuron the localiser ranks
most suspicious equals the detector) vs chance (1 / number of hidden neurons) and vs audit_baselines' stat_outlier, plus
detection AUROC (backdoored vs clean). Upper bound: the same localiser trained on val_audit backdoors.
