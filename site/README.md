# WeightGym showcase page

A static page that presents the WeightGym results. `index.html` is self-contained (data, CSS and JS
inlined); it loads nothing external except optional Google Fonts and falls back to system fonts offline.

## Rebuild

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python site/build_data.py
```

This reads `results/*.csv`, `runs/cost_tally.json`, run configs and rollout logs, writes `site/data.json`,
and renders `site/index.html` from `site/src/template.html`, `style.css` and `app.js`. It fails if:

- the template contains a typed-in percentage, dollar amount or decimal (prose numbers must be `{{fields}}`);
- `README.md` does not contain the same figures (54 checks: headline, every test split, comparison,
  verifier, improve loop, real data, costs including "supervised stage $123.50, RL $39.69", training
  settings, RL minus supervised, the progression table);
- re-scoring the replayed answers with `weightgym.env.score_answer`, or re-applying the replayed
  proposals, does not reproduce the logged outcome;
- a rate in `results/table_main.csv` disagrees with its rollout log;
- a disguised network (see below) does not compute the same outputs as the original on every input;
- `results/table_rl_vs_sft.csv` (used when present; otherwise the paired intervals are recomputed from the
  logs) has a point estimate that disagrees with the logs.

Rates are recomputed from the rollout logs (the CSVs keep 4 decimals, which rounds 269/403 = 66.749% up
to 66.75%), and 1/n-valued numbers from the other CSVs are snapped back to k/n before rounding.

## View locally

```bash
cd site && python -m http.server 8000     # then open http://localhost:8000
```

Opening `index.html` straight from disk also works. Add `?theme=dark` or `?theme=light` to force a theme.

## What each figure shows and where it comes from

| Figure | Source |
|---|---|
| Summary card (top) | Headline rates and tokens (`table_main.csv`, logs), validation curve (`runs/sft_scale`, `runs/rl_scale`), `cost_tally.json`, `table_improve.csv`. Exported as `media/summary-1600x900.png` and `media/og-1200x630.png` (the `og:image`). |
| Token race | Beat 1: `data/test_id.jsonl` network 3000218; left is the `thinking` text and answer of that seed in `runs/final/untrained_thinking/test_id.jsonl` (9,149 tokens). Beat 2: rows of `runs/final/scale_rl/test_id.jsonl` in file order, starting at that seed and wrapping, faulty networks only, added while the running sum of their logged `n_out` stays within 9,149. Result: 207 networks in 9,142 tokens, 176 repaired, 31 missed (tiles use the logged `success`). Same token count, not wall-clock time. |
| The repair | Network 3000218: weights text cut from `render_prompt(ep, direct=True)`, planted fault from `fault_info`, the trained model's edit, accuracy re-scored with `score_answer`. |
| Nothing to memorise | `sample_task("arith", default_rng([0, 11]), 0)` from `weightgym/nets.py`, then `permute_hidden`, `rescale_hidden` (c = 1.5, 0.8), `split_neuron` (0.4 : 0.6) and `add_dead_neuron`, cumulatively. Each variant gives the same outputs as the original on all 1,000 inputs (exact weights, max difference about 4e-15), and its 2-decimal weights (shown, rendered by `render_weights`) are correct on every input. |
| Six test sets | `scale_rl`, `untrained`, `baseline_outlier` rows of `results/table_main.csv`, rates recomputed from `runs/final/*/test_*.jsonl`. |
| Training on Tinker | `runs/sft_scale/metrics.jsonl` (`kind == "val"`) and `runs/rl_scale/chunks.jsonl`; selected chunk from the log's `best` field. |
| Tokens vs accuracy | `results/table_main.csv` (`mean_out_tokens`); frontier reasoning tokens from the frontier logs. Below 560 px the direct labels become numbered markers with a key. |
| Verifier | `results/table_verify.csv`. |
| Propose and verify | `data/test_imperfect.jsonl` network 42000123 and its 8 logged proposals in `runs/final_improve/rl_proposer/test_imperfect.jsonl`, re-applied at build time. |
| Improve curve | `results/table_improve.csv`. |
| Pure RL collapse | `runs/sweep/rl_lr*/metrics.jsonl` (`env/all/noop_on_faulty`). |
| Image classifiers | `results/table_imagehead.csv` (formula-trained proposer). |

The code excerpts are copied from `weightgym/tinker_train.py` at build time, with their line numbers.

## Qwen3.5-0.8B slot

The section "A real LLM: Qwen3.5-0.8B" (and its navigator link) is hidden unless `site/qwen.json` exists:

```json
{ "status": "...",
  "rows": [{"label": "...", "math500": 0.356, "lo": 0.30, "hi": 0.416, "n": 250, "note": "..."}],
  "bars": [{"label": "...", "value": 0.356, "lo": 0.30, "hi": 0.416, "note": "...", "highlight": true}],
  "caveats": ["..."],
  "summary": "..." }
```

`math500`, `value`, `lo` and `hi` are fractions. `bars` (optional) is drawn as a horizontal bar figure in the
style of the six-test-set chart (highlighted bars in the trained colour); `caveats` (optional) is shown as a
"Limits" list. Rerun `build_data.py` after writing the file to inline it (the build checks `rows` and
`summary`). If it is not inlined, the page still fetches `qwen.json` when served over HTTP. To test the slot
without creating the file: `WG_QWEN_JSON=/path/to/test.json` on the build command.

## Video clips and stills

`site/media/`: `race`, `splits`, `repair`, `loop` and `disguise` as `.mp4` (1280x720, H.264, 7-17 s) and
`.gif` (960 px, 12 fps), plus `summary-1600x900.png` and `og-1200x630.png`. They are frames of the page's own
animations (`index.html?clip=<name>`), rendered with headless Chromium and encoded with ffmpeg:

```bash
source site/tools/env.sh
$TOOLS_PY site/tools/export_clips.py          # all clips and stills, or name them (card = stills)
$TOOLS_PY site/tools/shoot.py OUT_DIR         # desktop/phone, light/dark screenshots
$TOOLS_PY site/tools/check_numbers.py         # every %, $ and decimal on the rendered page traced to data.json
```

The toolchain lives in `site/.tools/` (git-ignored): a venv with Playwright, Pillow and imageio-ffmpeg, the
Playwright Chromium build, and three system libraries (libnss3, libnspr4, libasound2) extracted from Ubuntu
packages without root.

## Byline

Set once, in `BYLINE` at the top of `build_data.py`. `REPO_URL` there is a placeholder for the public
repository link and is not used on the page yet.
