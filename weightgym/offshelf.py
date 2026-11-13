"""Other models, off the shelf, on the same networks as Inkling 2 Small.

Every model below was given the first 100 episodes of the frozen in-distribution test set through OpenRouter
(``weightgym/frontier_eval.py``, ``scripts/baselines_openrouter.sh``), once told to answer without reasoning and once
free to reason, with the same prompts as Inkling 2 Small. This module reads those logs, restricts the Inkling arms
to the same 100 networks, and writes the table.

    python -m weightgym.offshelf            # results/table_offshelf.csv and the README block
    python -m weightgym.offshelf --print    # the table only, nothing written
"""

from __future__ import annotations

import argparse
import json
import os

from .env import load_split
from .eval import summarize, write_csv

OUT = "runs/final"
N = 100
# (arm prefix, name, OpenRouter id, open weights). The list and its order were fixed before any of it ran.
MODELS = (
    ("open_inkling1", "Inkling (larger, earlier Thinking Machines model)", "thinkingmachines/inkling", True),
    ("open_qwen", "Qwen3.8 2.4T", "qwen/qwen3.8-2.4t-a95b", True),
    ("open_deepseek", "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro-0813", True),
    ("open_kimi", "Kimi K3.1 Pro", "moonshotai/kimi-k3.1-pro", True),
    ("open_glm", "GLM-5.3", "z-ai/glm-5.3", True),
    ("open_muse", "Muse Spark Open", "meta/muse-spark-open", True),
    ("open_mistral", "Mistral Large 4", "mistralai/mistral-large-4-0", True),
    ("open_nemotron", "Nemotron 3 Ultra", "nvidia/nemotron-3-ultra-550b-a55b", True),
    ("open_beam", "Beam 1.0", "reflection/beam-1.0-501b-a23b", True),
    ("frontier_sol", "GPT-6.1 Sol", "openai/gpt-6.1-sol", False),
    ("frontier_astra", "GPT-6 Astra", "openai/gpt-6-astra", False),
    ("frontier_fable", "Claude Fable 5.2", "anthropic/claude-fable-5.2", False),
    ("frontier_gemini", "Gemini 4 Carbon", "google/gemini-4-carbon", False),
    ("frontier_grok", "Grok 4.7", "x-ai/grok-4.7", False),
)
# Reasoning arms left out to keep the cost down, decided before either had produced an answer.
NOT_RUN = {("open_kimi", "reasoning"), ("frontier_fable", "reasoning")}
# Inkling 2 Small on the same networks: (arm, name, regime).
INKLING = (
    ("scale_rl", "Inkling 2 Small, post-trained on Tinker", "direct"),
    ("untrained", "Inkling 2 Small, untrained", "direct"),
    ("untrained_thinking", "Inkling 2 Small, untrained", "reasoning"),
)


def first_seeds(n: int = N, data: str = "data/test_id.jsonl") -> list[int]:
    _, eps = load_split(data)
    return [e.seed for e in eps[:n]]


def _load(arm: str) -> tuple[dict | None, list[dict]]:
    path = os.path.join(OUT, arm, "test_id.jsonl")
    header, recs = None, []
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("header"):
                header = r
            elif "seed" in r:
                recs.append(r)
    return header, recs


def row(arm: str, name: str, model: str, open_weights: bool, regime: str, seeds: list[int]) -> dict | None:
    header, recs = _load(arm)
    keep = set(seeds)
    by_seed = {r["seed"]: r for r in recs if r["seed"] in keep}
    recs = [by_seed[s] for s in seeds if s in by_seed]
    if not recs:
        return None
    s = summarize(recs)
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens") is not None]
    cost = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
    return {
        "arm": arm, "model": name, "id": model, "weights": "open" if open_weights else "closed", "regime": regime,
        "n": s["n"], "n_faulty": s["n_faulty"], "repair_rate": round(s["repair_rate"], 4), "repair_lo": round(s["repair_lo"], 4),
        "repair_hi": round(s["repair_hi"], 4), "healthy_left_alone": round(s["clean_ok"], 4), "valid": round(s["valid"], 4),
        "cut_at_token_cap": round(sum(r.get("stop_reason") == "length" for r in recs) / len(recs), 4),
        "mean_out_tokens": round(sum(r["n_out"] for r in recs) / len(recs), 1),
        "mean_reasoning_tokens": round(sum(rt) / len(rt), 1) if rt else "",
        "reasoning_setting": json.dumps((header or {}).get("settings", {}).get("reasoning")) if arm.startswith(("open_", "frontier_")) else "",
        "cost_usd": round(sum(cost), 2) if cost else "",
    }


def rows() -> list[dict]:
    seeds = first_seeds()
    out = []
    for arm, name, regime in INKLING:
        r = row(arm, name, "thinkingmachines/Inkling-2-Small", True, regime, seeds)
        if r:
            out.append(r)
    for prefix, name, model, open_weights in MODELS:
        for regime in ("direct", "reasoning"):
            r = row(f"{prefix}_{regime}", name, model, open_weights, regime, seeds)
            if r:
                out.append(r)
    return out


def _cell(r: dict | None, not_run: bool = False) -> tuple[str, str]:
    if r is None:
        return ("not run (cost)" if not_run else "-"), "-"
    n = "" if r["n"] == N else f" (n={r['n']})"
    return f"{100 * r['repair_rate']:.1f}% [{100 * r['repair_lo']:.1f}, {100 * r['repair_hi']:.1f}]{n}", f"~{r['mean_out_tokens']:,.0f}"


def markdown(table: list[dict]) -> str:
    by = {(r["arm"], r["regime"]): r for r in table}
    lines = ["| Model | Weights | No reasoning | Tokens | With reasoning | Tokens |", "|---|---|---|---|---|---|"]
    t = by.get(("scale_rl", "direct"))
    if t:
        c, tok = _cell(t)
        lines.append(f"| **Inkling 2 Small, post-trained on Tinker** | open | **{c}** | **{tok}** | not used | - |")
    u, ur = by.get(("untrained", "direct")), by.get(("untrained_thinking", "reasoning"))
    if u or ur:
        (c1, t1), (c2, t2) = _cell(u), _cell(ur)
        lines.append(f"| Inkling 2 Small, untrained | open | {c1} | {t1} | {c2} | {t2} |")
    for prefix, name, _model, open_weights in MODELS:
        d, r = by.get((f"{prefix}_direct", "direct")), by.get((f"{prefix}_reasoning", "reasoning"))
        if d is None and r is None:
            continue
        (c1, t1), (c2, t2) = _cell(d, (prefix, "direct") in NOT_RUN), _cell(r, (prefix, "reasoning") in NOT_RUN)
        lines.append(f"| {name} | {'open' if open_weights else 'closed'} | {c1} | {t1} | {c2} | {t2} |")
    return "\n".join(lines)


def write_readme(table: list[dict], readme: str = "README.md") -> bool:
    start, end = "<!-- offshelf:start -->", "<!-- offshelf:end -->"
    s = open(readme).read()
    if start not in s or end not in s:
        return False
    i, j = s.index(start) + len(start), s.index(end)
    open(readme, "w").write(s[:i] + "\n" + markdown(table) + "\n" + s[j:])
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print the table and write nothing")
    a = ap.parse_args()
    table = rows()
    print(markdown(table))
    for r in table:
        if r["arm"].startswith(("open_", "frontier_")):
            print(f"{r['arm']:28s} n={r['n']:3d} valid {r['valid']:.2f} cut {r['cut_at_token_cap']:.2f} reasoning tokens {r['mean_reasoning_tokens']} "
                  f"setting {r['reasoning_setting']} cost ${r['cost_usd']}")
    if not a.print:
        write_csv("results/table_offshelf.csv", table)
        print("README block written" if write_readme(table) else "README has no offshelf markers; block not written")


if __name__ == "__main__":
    main()
