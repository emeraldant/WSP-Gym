# Literature check (weight-space programming, Nov 2026) — summary used to plan round 3
Read by web-fetch agents; some sections were model-summarised, numbers are as reported, not re-verified.

- arXiv 2611.21449 (Goworek et al., Turing Inst., Nov 16): replication. GPT-6 Astra flagged training-time poison 10/10,
  fine-tuning poison 7/10, malicious persona adapter 23/32 from plain-text weights, but never found the trigger.
  Decompile-and-compare (49,189 states) as a check that a reading is real.
- arXiv 2610.07334 Weight Oracles (Kabra, Venhoff, Schroeder de Witt, Oxford, Oct 5): Qwen 2.5 3B with weight rows as
  continuous tokens; single pass stalls at 45% (shortcut), 5 chained passes reach 99%. Trained on 10 benign pathologies
  only, detects unseen backdoors AUROC 0.93 (attention-routed) / 0.81 (diverse). ~2K-param models, 20% FPR at 94% TPR,
  single-token triggers.
- arXiv 2611.18553 WeightBench (Nov 13): 42 tasks; infeasibility track; metered forward passes; GPT-6.1 Sol 0.568,
  Opus 5.5 0.546 on 38 tasks.
- arXiv 2611.08070 (Nov 5): Opus 5.5 names datasets from CNN weights partly from priors (CIFAR-10 at 45% confidence on
  shuffled weights; all class pairs wrong under label permutation).
- LessWrong replication of Sparks (Nov 8): DeepSeek-V4.1-Flash names a game for untrained weights 14/60, Pac-Man prior;
  control suite: shuffled / untrained / anonymised / random-subset weights.
- arXiv 2611.18114 iWSP (Nov 13): agent writes weights in code; LLM auditor; rebuild-from-200-word-description test.
- arXiv 2611.22055 (Nov 17): associative-memory nets as WSP targets; single-line removal changes 0.5-1.8 answers (locality).
- arXiv 2611.09382 (Nov 6): Astra edits Qwen3.5; +29.45 MATH-500; sign-reversed 0/48 and norm-matched random 10/48 controls
  vs targeted 20/48 (base 6/48); 9B dev gain +29.69 became -0.35 held-out.
- arXiv 2611.22228 (Nov 17): score-guided rewrite raises proxy, lowers task success (proxy/task divergence).
