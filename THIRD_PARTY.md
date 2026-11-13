# Third-party material

The MIT License in `LICENSE` covers the code, the generated networks and the logs written for this project.
The following stay under their own terms.

- **Answer extraction in `weightgym/qwen_bench.py`** (`last_boxed`, `_box_at`, `answer_text`) follows the grader
  released with "Sparks of Weight-Space Programming" (`exp8_math/code/mathqa.py` in
  github.com/RikiyaT/lm_understand-logs), MIT License, Copyright (c) 2026 the authors of "LLMs can understand
  and write neural network weights", which in turn follows hendrycks/math (MIT License).
- **Tabular datasets** behind `data/real_*.jsonl` come from OpenML and UCI; each dataset's name, source and
  licence is listed in `data/realworld_datasets.json`.
- **Public models**: `data/real_hf_models.jsonl` holds the weights of
  `bernardozschaber/titanic-federated-dp-mlp` (MIT). The image classifiers in `data/imagehead_models.jsonl` are
  referenced by their Hugging Face Hub ids and are not redistributed here.
- **Qwen3.5-0.8B** (Apache-2.0) is downloaded from the Hub by the scripts and is not redistributed. MATH-500,
  GSM8K and MATH problems are loaded from the Hub at run time; the logs under `runs/qwen*` contain model outputs,
  not the problem sets.
- **tinker-cookbook** (Thinking Machines) provides the RL loop used by `weightgym/tinker_train.py`.
