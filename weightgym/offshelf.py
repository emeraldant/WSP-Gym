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
from decimal import ROUND_HALF_UP, Decimal

from .env import load_split
from .eval import paired_diff_ci, summarize, write_csv

OUT = "runs/final"
N = 100
# (arm prefix, name, OpenRouter id, open weights). The list and its order were fixed before any of it ran.
MODELS = (
    ("open_inkling1", "Inkling (first generation, larger)", "thinkingmachines/inkling", True),
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
# Added on Nov 13 after the list above had been scored, at the no-reasoning setting only.
ADDED = (
    ("frontier_opus", "Claude Opus 5.5", "anthropic/claude-opus-5.5", False),
    ("frontier_sonnet", "Claude Sonnet 5.5", "anthropic/claude-sonnet-5.5", False),
    ("frontier_haiku", "Claude Haiku 5.5", "anthropic/claude-haiku-5.5", False),
    ("frontier_hydrogen", "Gemini 4 Hydrogen", "google/gemini-4-hydrogen", False),
    ("frontier_luna", "GPT-6 Luna", "openai/gpt-6-luna", False),
)
# Arms that were not run, and why. The two cost ones were left out before either had produced an answer.
NOT_RUN = {("open_kimi", "reasoning"): "not run (cost)", ("frontier_fable", "reasoning"): "not run (cost)",
           **{(m[0], "reasoning"): "not run" for m in ADDED}}
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
    if not rt and not any((r.get("thinking") or "").strip() for r in recs):
        rt = [0] * len(recs)  # Tinker logs carry the reasoning text, not a count: no text in any answer means none
    cost = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
    # Faulty networks whose answer was not cut off at the token limit, and the share of those repaired. Descriptive
    # only: the networks a model finishes are not a random sample of the networks it was given.
    finished = [r for r in recs if not r["clean"] and r.get("stop_reason") != "length"]
    fin = {"finished_faulty": len(finished),
           "fixed_of_finished": round(sum(r["success"] for r in finished) / len(finished), 6) if finished else ""}
    return {
        **fin,
        "arm": arm, "model": name, "id": model, "weights": "open" if open_weights else "closed", "regime": regime,
        "n": s["n"], "n_faulty": s["n_faulty"], "repair_rate": round(s["repair_rate"], 6), "repair_lo": round(s["repair_lo"], 6),
        "repair_hi": round(s["repair_hi"], 6), "healthy_left_alone": round(s["clean_ok"], 4), "valid": round(s["valid"], 4),
        "cut_at_token_cap": round(sum(r.get("stop_reason") == "length" for r in recs) / len(recs), 4),
        "mean_out_tokens": round(sum(r["n_out"] for r in recs) / len(recs), 1),
        "mean_reasoning_tokens": round(sum(rt) / len(rt), 1) if rt else "",
        "no_reasoning_share": round(sum(x == 0 for x in rt) / len(rt), 4) if rt else "",
        "reasoning_setting": json.dumps((header or {}).get("settings", {}).get("reasoning")) if arm.startswith(("open_", "frontier_")) else "",
        "cost_usd": round(sum(cost), 2) if cost else "", "added_later": 0,
    }


def _success(arm: str, seeds: list[int]) -> dict[int, float]:
    keep = set(seeds)
    return {r["seed"]: r["success"] for r in _load(arm)[1] if r["seed"] in keep and not r["clean"]}


def _against_trained(r: dict, seeds: list[int]) -> dict:
    """Trained Inkling 2 Small (no reasoning) minus this arm, paired on the faulty networks both answered."""
    a, b = _success("scale_rl", seeds), _success(r["arm"], seeds)
    common = sorted(set(a) & set(b))
    if r["arm"] == "scale_rl" or not common:
        return {"trained_minus_this": "", "diff_lo": "", "diff_hi": "", "diff_n": ""}
    d, lo, hi = paired_diff_ci([a[s] for s in common], [b[s] for s in common])
    return {"trained_minus_this": round(d, 6), "diff_lo": round(lo, 6), "diff_hi": round(hi, 6), "diff_n": len(common)}


def rows() -> list[dict]:
    out = _rows()
    seeds = first_seeds()
    return [{**r, **_against_trained(r, seeds)} for r in out]


def _rows() -> list[dict]:
    seeds = first_seeds()
    out = []
    for arm, name, regime in INKLING:
        r = row(arm, name, "thinkingmachines/Inkling-2-Small", True, regime, seeds)
        if r:
            out.append(r)
    for prefix, name, model, open_weights in MODELS + ADDED:
        for regime in ("direct", "reasoning"):
            r = row(f"{prefix}_{regime}", name, model, open_weights, regime, seeds)
            if r:
                r["added_later"] = int((prefix, name, model, open_weights) in ADDED)
                out.append(r)
    return out


def tokens_cell(r: dict, bracket: bool) -> str:
    """Mean tokens generated per network; with ``bracket``, how many of them were reasoning (where that is known)."""
    rt = r.get("mean_reasoning_tokens", "")
    return f"~{float(r['mean_out_tokens']):,.0f}" + (f" ({float(rt):,.0f})" if bracket and rt != "" else "")


def _cell(r: dict | None, not_run: str = "", bracket: bool = False) -> tuple[str, str, str]:
    if r is None:
        return not_run or "-", "-", "-"
    n = "" if int(float(r["n"])) == N else f" (n={int(float(r['n']))})"
    return (f"{_pct(r['repair_rate'])} [{_pct(r['repair_lo'])[:-1]}, {_pct(r['repair_hi'])[:-1]}]{n}", _pct(r["cut_at_token_cap"], 0),
            tokens_cell(r, bracket))


def markdown(table: list[dict]) -> str:
    by = {(r["arm"], r["regime"]): r for r in table}
    lines = ["| Model | Weights | Told to answer at once: repaired | cut off | tokens (of them reasoning) | Free to reason: repaired | cut off | tokens |",
             "|---|---|---|---|---|---|---|---|"]
    t = by.get(("scale_rl", "direct"))
    if t:
        c, cut, tok = _cell(t, bracket=True)
        lines.append(f"| **Inkling 2 Small, post-trained on Tinker** | open | **{c}** | {cut} | **{tok}** | not used | - | - |")
    u, ur = by.get(("untrained", "direct")), by.get(("untrained_thinking", "reasoning"))
    if u or ur:
        lines.append(f"| Inkling 2 Small, untrained | open | {' | '.join(_cell(u, bracket=True))} | {' | '.join(_cell(ur))} |")
    for prefix, name, _model, open_weights in MODELS + ADDED:
        d, r = by.get((f"{prefix}_direct", "direct")), by.get((f"{prefix}_reasoning", "reasoning"))
        if d is None and r is None:
            continue
        star = " \\*" if (prefix, name, _model, open_weights) in ADDED else ""
        lines.append(f"| {name}{star} | {'open' if open_weights else 'closed'} | {' | '.join(_cell(d, NOT_RUN.get((prefix, 'direct'), ''), True))} "
                     f"| {' | '.join(_cell(r, NOT_RUN.get((prefix, 'reasoning'), '')))} |")
    return "\n".join(lines)


def _pct(x, d: int = 1) -> str:
    """Round half up, the way the page does, so the README and the page print the same figure."""
    q = Decimal(1).scaleb(-d)
    return f"{Decimal(str(round(100 * float(x), 6))).quantize(q, rounding=ROUND_HALF_UP)}%"


def _joined(xs) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def reading(table: list[dict]) -> dict[str, str]:
    """The table in sentences, every figure taken from its rows (which may be read back from the CSV as text)."""
    f = float
    by = {(r["arm"], r["regime"]): r for r in table}
    names = {m[0]: m[1] for m in MODELS + ADDED}
    nm = lambda r: names[r["arm"].rsplit("_", 1)[0]]
    others = [r for r in table if r["arm"].startswith(("open_", "frontier_"))]
    sel = lambda regime, w: [r for r in others if r["regime"] == regime and r["weights"] == w]
    tok = lambda x: f"{f(x):,.0f}"

    def span(rs, key="repair_rate", fmt=_pct):
        v = [f(r[key]) for r in rs]
        return fmt(min(v)) if fmt(min(v)) == fmt(max(v)) else f"{fmt(min(v))} to {fmt(max(v))}"

    trained, think = by[("scale_rl", "direct")], by[("untrained_thinking", "reasoning")]
    out = {"n": str(N), "faulty": str(int(f(trained["n_faulty"]))), "trained": _pct(trained["repair_rate"]),
           "trained_tokens": tok(trained["mean_out_tokens"]), "think": _pct(think["repair_rate"])}

    # A model "answered at once" if fewer than a quarter of its answers ran past the 512-token limit and it did not
    # reason at length outside it (more than 1,000 tokens generated per network).
    CUT = 0.25
    open_d, closed_d = sel("direct", "open"), sel("direct", "closed")
    heavy = [r for r in open_d + closed_d if f(r["mean_out_tokens"]) > 1000 and f(r["cut_at_token_cap"]) < CUT]
    stuck = [r for r in open_d + closed_d if f(r["cut_at_token_cap"]) >= CUT]
    once = [r for r in open_d + closed_d if r not in heavy and r not in stuck]
    once_open, once_closed = [r for r in once if r["weights"] == "open"], [r for r in once if r["weights"] == "closed"]
    ahead = sorted((r for r in once_closed if f(r["diff_hi"]) < 0), key=lambda r: -f(r["repair_rate"]))
    rest = [r for r in once_closed if r not in ahead]
    best = max(once_open, key=lambda r: f(r["repair_rate"]))
    out.update(n_models=str(len(open_d + closed_d)), n_once=str(len(once)), n_once_open=str(len(once_open)),
               best_open=_pct(best["repair_rate"]), best_open_name=nm(best), n_stuck=str(len(stuck)))
    # Of those, which really had reasoning switched off, and which only ran at their API's lowest setting.
    is_off = lambda r: json.loads(r.get("reasoning_setting") or "null") == {"enabled": False}
    off, soft = [r for r in once if is_off(r)], [r for r in once if not is_off(r)]
    assert all(f(r["mean_reasoning_tokens"]) == 0 for r in off) and f(trained["mean_reasoning_tokens"]) == 0
    off_open = [r for r in off if r["weights"] == "open"]
    assert once_open == off_open, "the page's summary line says the open models that answered at once had reasoning off"
    parts = [f"Told to answer at once, {len(once)} of the {len(open_d + closed_d)} models did, in the sense that fewer than a quarter of their "
             f"answers ran past the 512-token limit. {len(off)} of them ({len(off_open)} open-weights, {len(off) - len(off_open)} closed) ran with "
             f"reasoning switched off and repair {span(off)}; the trained Inkling 2 Small repairs {out['trained']} in about "
             f"{out['trained_tokens']} tokens, also with no reasoning."]
    if soft:
        glance = sorted((r for r in soft if f(r["no_reasoning_share"]) >= 0.9), key=lambda r: -f(r["repair_rate"]))
        brief = sorted((r for r in soft if r not in glance), key=lambda r: -f(r["repair_rate"]))
        c = [f"{nm(r)} still used no reasoning tokens on {_pct(r['no_reasoning_share'], 0)} of networks and repairs {_pct(r['repair_rate'])}" for r in glance]
        c += [f"{nm(r)} repairs {_pct(r['repair_rate'])} after about {tok(r['mean_reasoning_tokens'])} reasoning tokens per network" for r in brief]
        parts.append(f"The other {len(soft)} cannot have reasoning switched off through their API and ran at its lowest setting: " + "; ".join(c) + ".")
    for r in heavy:
        parts.append(f"{nm(r)} repairs {_pct(r['repair_rate'])} after about {tok(r['mean_out_tokens'])} tokens of reasoning, which its API does not count against the limit.")
    out["direct"] = " ".join(parts)
    if stuck:
        well = sorted((r for r in stuck if f(r["finished_faulty"]) >= 20 and f(r["fixed_of_finished"]) >= 0.9), key=lambda r: -f(r["repair_rate"]))
        out["stuck"] = (
            f"The other {len(stuck)} models kept reasoning, in reasoning tokens or in the answer itself, and were cut off on "
            f"{span(stuck, 'cut_at_token_cap', lambda x: _pct(x, 0))} of networks. "
            f"Their figure in that column ({span(stuck)}) says how often their reasoning fits in 512 tokens, not how well they answer at a glance"
            + (f": {_joined(nm(r) for r in well)}, for example, repair {span(well, 'fixed_of_finished', lambda x: _pct(x, 0))} of the faulty networks they finish."
               if well else "."))
    else:
        out["stuck"] = ""

    # With reasoning. The limit decides most rows, so say how often each model finishes as well as how often it repairs.
    open_r, closed_r = sel("reasoning", "open"), sel("reasoning", "closed")
    fin = lambda r: 1 - f(r["cut_at_token_cap"])
    finspan = lambda rs: span([{"x": fin(r)} for r in rs], "x", lambda x: _pct(x, 0))
    level = [r for r in open_r if f(r["diff_lo"]) <= 0]
    below = [r for r in open_r if r not in level]
    rated = lambda rs: _joined(f"{nm(r)} ({_pct(r['repair_rate'])})" for r in sorted(rs, key=lambda r: -f(r["repair_rate"])))
    allr = open_r + closed_r + [think]
    parts = [f"Free to reason, the 20,000-token limit decides most rows: every model repairs {span(allr, 'fixed_of_finished', lambda x: _pct(x, 0))} "
             f"of the faulty networks it finishes. The networks a model finishes are probably the easier ones, so that is a guide to what a "
             f"higher limit would give, not a measurement of it."]
    out["tldr_reason"] = ""
    if level:
        ratios = sorted(round(f(r["mean_out_tokens"]) / f(trained["mean_out_tokens"]), -1) for r in level)
        times = f"about {tok(ratios[0])}" if ratios[0] == ratios[-1] else f"{tok(ratios[0])} to {tok(ratios[-1])}"
        parts.append(f"{rated(level)} {'finishes' if len(level) == 1 else 'finish'} {finspan(level)} of networks and "
                     f"{'is' if len(level) == 1 else 'are'} level with the trained model, at {times} times the tokens.")
        out["tldr_reason"] = (f"given 20,000 tokens to reason, {_joined(nm(r) for r in sorted(level, key=lambda r: -f(r['repair_rate'])))} "
                              f"{'matches' if len(level) == 1 else 'match'} the trained model at {times} times the tokens")
    if below:
        parts.append(f"The {'other ' if level else ''}{len(below)} open models finish {finspan(below)} and repair {span(below)}.")
    parts.append(f"Untrained Inkling 2 Small, at the low reasoning effort used for it throughout, finishes {_pct(fin(think), 0)} and repairs {out['think']}"
                 + (f"; the {len(closed_r)} closed models finish {finspan(closed_r)} and repair {span(closed_r)}." if closed_r else "."))
    out["reasoning"] = " ".join(parts)

    # The models that answered at once against the trained model, paired on the same faulty networks.
    pts = lambda r: f"{-100 * f(r['trained_minus_this']):+.1f} points [{-100 * f(r['diff_hi']):+.1f}, {-100 * f(r['diff_lo']):+.1f}]"
    p_ahead = sorted((r for r in once if f(r["diff_hi"]) < 0), key=lambda r: f(r["trained_minus_this"]))
    p_tie = [r for r in once if f(r["diff_lo"]) <= 0 <= f(r["diff_hi"])]
    p_behind = [r for r in once if f(r["diff_lo"]) > 0]
    closest = min(p_behind, key=lambda r: f(r["trained_minus_this"]))
    out["paired"] = (
        f"Paired on the same {out['faulty']} faulty networks, among the models that answered at once: "
        + (f"{_joined(f'{nm(r)} ({pts(r)})' for r in p_ahead)} {'is' if len(p_ahead) == 1 else 'are'} ahead of the trained Inkling 2 Small"
           if p_ahead else "none is ahead of the trained Inkling 2 Small")
        + (f", {_joined(nm(r) for r in p_tie)} cannot be told apart from it" if p_tie else "")
        + f", and the other {len(p_behind)} are behind it (the closest, {nm(closest)}, by {100 * f(closest['trained_minus_this']):.1f} points "
        f"[{100 * f(closest['diff_lo']):.1f}, {100 * f(closest['diff_hi']):.1f}]).")
    out["method"] = (
        f"Each model was given the first {N} networks of the in-distribution test set ({out['faulty']} of them faulty) through OpenRouter, "
        f"with the same two prompts as Inkling 2 Small: told to answer at once (the lowest reasoning setting its API accepts, 512-token "
        f"limit), then free to reason (20,000-token limit).")
    out["note"] = (
        "An answer cut off at the limit counts as not repaired, for every model including Inkling 2 Small; \"cut off\" is the share of the "
        f"{N} networks where that happened. Tokens are the mean generated per network, reasoning included; in the answer-at-once column the "
        "figure in brackets is how many of them were reasoning. Where an API would not switch reasoning off, the model ran at its lowest "
        "setting. The limits were chosen for "
        "Inkling 2 Small (512 tokens is ample for an answer, and its own limit there is 256; 20,000 is where its reasoning mostly finishes at "
        "the effort used), and the other "
        "models reasoned at their provider's default effort, so the with-reasoning column compares models at a fixed budget, not at their best. "
        "\\* Added after the first results, answer-at-once only. Two with-reasoning runs were left out for cost, before either had produced an answer.")
    return out


def write_readme(table: list[dict], readme: str = "README.md") -> bool:
    start, end = "<!-- offshelf:start -->", "<!-- offshelf:end -->"
    s = open(readme).read()
    if start not in s or end not in s:
        return False
    i, j = s.index(start) + len(start), s.index(end)
    r = reading(table)
    block = (f"{r['method']}\n\n{r['direct']}{' ' + r['stuck'] if r['stuck'] else ''}\n\n{r['reasoning']}\n\n{markdown(table)}\n\n"
             f"{r['note']}\n\n{r['paired']}")
    open(readme, "w").write(s[:i] + "\n" + block + "\n" + s[j:])
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print the table and write nothing")
    a = ap.parse_args()
    table = rows()
    print(markdown(table))
    for k, v in reading(table).items():
        print(f"[{k}] {v}")
    for r in table:
        if r["arm"].startswith(("open_", "frontier_")):
            print(f"{r['arm']:28s} n={r['n']:3d} valid {r['valid']:.2f} cut {r['cut_at_token_cap']:.2f} reasoning tokens {r['mean_reasoning_tokens']} "
                  f"setting {r['reasoning_setting']} cost ${r['cost_usd']}")
    if not a.print:
        write_csv("results/table_offshelf.csv", table)
        print("README block written" if write_readme(table) else "README has no offshelf markers; block not written")


if __name__ == "__main__":
    main()
