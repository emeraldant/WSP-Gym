"""Build the showcase page's data from the repository's results and logs.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python site/build_data.py

Reads results/*.csv, runs/cost_tally.json, run configs and rollout logs; writes site/data.json and
renders site/index.html from site/src/ (template, CSS, JS) with the data inlined, so the page also
works when opened straight from disk.

Every number on the page comes from here. Prose numbers are template fields ({{name}}) filled from
the data; a lint step fails the build if the template contains a typed-in percentage, dollar amount
or decimal. A second check fails if README.md does not contain the same figures as the data, so the
page and the write-up cannot drift apart silently.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE = os.path.join(ROOT, "site")
sys.path.insert(0, ROOT)

from weightgym.edits import apply_edit, validate_edit  # noqa: E402
from weightgym.env import load_split, render_prompt, score_answer  # noqa: E402
from weightgym.eval import paired_diff_ci, summarize  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Settings that are choices, not results
# ------------------------------------------------------------------------------------------------

BYLINE = "@emeraldant"  # the only place the byline is set
BYLINE_URL = "https://x.com/emeraldant"
REPO_URL = "https://github.com/emeraldant/WSP-Gym"

HERO_SEED = 3000218  # test_id network replayed in the hero race and the repair figure (see site/README.md)
IMPROVE_SEED = 42000123  # test_imperfect network replayed in the propose-and-verify figure (gap closed near the mean among improved networks)

SPLITS = [  # (split, short label, long label)
    ("test_id", "In-distribution", "In-distribution"),
    ("test_larger", "Larger networks", "Larger networks than any in training"),
    ("test_families", "Unseen families", "Function families never trained on"),
    ("test_faults", "Unseen fault types", "Fault types never trained on"),
    ("test_trained", "Gradient-made", "Networks made by gradient descent (our trainer)"),
    ("test_external", "scikit-learn", "Networks fitted by scikit-learn (an optimiser never used in training)"),
]

CHECKS: list[str] = []  # human-readable log of what was verified


def p(*parts) -> str:
    return os.path.join(ROOT, *parts)


def read_csv(name: str) -> list[dict]:
    with open(p("results", name)) as f:
        return list(csv.DictReader(f))


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def records(path: str) -> list[dict]:
    return [r for r in read_jsonl(path) if not r.get("header")]


# ------------------------------------------------------------------------------------------------
# Formatting (one place, so the page and the README check use the same rounding)
# ------------------------------------------------------------------------------------------------


def _r(x: float, d: int, scale: float = 100.0) -> Decimal:
    """Round half up, as a person reading the 4-decimal CSVs would (0.0425 -> 4.3, not 4.2)."""
    q = Decimal(1).scaleb(-d)
    return Decimal(str(round(scale * float(x), 6))).quantize(q, rounding=ROUND_HALF_UP)


def snap(x, n) -> float:
    """Rates and bootstrap quantiles over n networks are multiples of 1/n; the CSVs store them rounded to
    4 decimals. Snapping back to k/n recovers the exact value so it rounds the same way as the write-up."""
    n = int(float(n))
    return round(float(x) * n) / n


def pct(x: float, d: int = 1) -> str:
    return f"{_r(x, d)}%"


def ci(lo: float, hi: float, d: int = 1) -> str:
    return f"[{_r(lo, d)}, {_r(hi, d)}]"


def _signed(v: Decimal) -> str:
    return f"+{v}" if v >= 0 else f"{v}"


def spts(x: float, d: int = 1) -> str:
    return _signed(_r(x, d))


def sci(lo: float, hi: float, d: int = 1) -> str:
    return f"[{_signed(_r(lo, d))}, {_signed(_r(hi, d))}]"


def gap(x: float, d: int = 1) -> str:  # share of errors removed, signed
    return f"{_signed(_r(x, d))}%"


def usd0(x: float) -> str:
    return f"${_r(x, 0, 1.0):,}"


def usd2(x: float) -> str:
    return f"${_r(x, 2, 1.0):,}"


def thousands(x: float, to: int = 100) -> str:
    return f"{int(round(float(x) / to) * to):,}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# ------------------------------------------------------------------------------------------------
# Results tables
# ------------------------------------------------------------------------------------------------


def main_table() -> dict:
    out: dict = defaultdict(dict)
    for r in read_csv("table_main.csv"):
        row = {}
        for k, v in r.items():
            if k in ("arm", "split"):
                continue
            try:
                row[k] = float(v) if v not in ("", "nan") else None
            except ValueError:
                row[k] = v
        out[r["arm"]][r["split"]] = row
    # The CSV keeps 4 decimals, which is not enough to round some rates correctly (269/403 = 0.66749 is
    # stored as 0.6675). Recompute every rate from its rollout log and check it against the CSV.
    for arm, splits in out.items():
        for split, row in splits.items():
            path = p("runs/final", arm, f"{split}.jsonl")
            if not os.path.exists(path):
                continue
            s = summarize(records(path))
            for k in ("repair_rate", "repair_lo", "repair_hi", "clean_ok", "clean_lo", "clean_hi"):
                if row.get(k) is not None and s[k] == s[k]:
                    assert abs(row[k] - s[k]) < 6e-5, (arm, split, k, row[k], s[k])
                    row[k] = s[k]
    CHECKS.append("every rate taken from results/table_main.csv was recomputed from its rollout log and agrees to 4 decimals")
    return out


def build() -> tuple[dict, dict]:
    M = main_table()
    T: dict[str, str] = {}  # template fields: display strings
    D: dict = {"byline": BYLINE, "byline_url": BYLINE_URL, "repo_url": REPO_URL}

    # --- headline -------------------------------------------------------------------------------
    rl, think, unt = M["scale_rl"]["test_id"], M["untrained_thinking"]["test_id"], M["untrained"]["test_id"]
    vs_r = next(r for r in read_csv("table_vs_reasoning.csv") if r["arm"] == "scale_rl" and r["vs"] == "untrained_thinking")
    vs_u = next(r for r in read_csv("table_paired_diff.csv") if r["arm"] == "scale_rl" and r["vs"] == "untrained" and r["split"] == "test_id")
    for row in (vs_r, vs_u):
        for k in ("diff", "lo", "hi"):
            row[k] = snap(row[k], row["n_faulty"])
    think_recs = records(p("runs/final/untrained_thinking/test_id.jsonl"))
    cap_share = float(np.mean([r["stop_reason"] == "length" for r in think_recs]))
    think_done = [r for r in think_recs if not r["clean"] and r["stop_reason"] != "length"]
    think_finished = float(np.mean([bool(r["success"]) for r in think_done]))  # repaired, of the faulty networks it finished
    think_cap_tokens = read_jsonl(p("runs/final/untrained_thinking/test_id.jsonl"))[0]["settings"]["max_tokens"]
    think_effort = read_jsonl(p("runs/final/untrained_thinking/test_id.jsonl"))[0]["settings"]["effort"]
    rl_settings = read_jsonl(p("runs/final/scale_rl/test_id.jsonl"))[0]["settings"]
    assert abs(float(vs_r["reasoning_rate"]) - think["repair_rate"]) < 1e-4  # both runs cover all 422 faulty networks
    D["headline"] = {
        "trained": {"rate": rl["repair_rate"], "lo": rl["repair_lo"], "hi": rl["repair_hi"], "tokens": rl["mean_out_tokens"], "n_faulty": rl["n_faulty"]},
        "reasoning": {"rate": think["repair_rate"], "lo": think["repair_lo"], "hi": think["repair_hi"], "tokens": think["mean_out_tokens"],
                      "cap_share": cap_share, "cap_tokens": think_cap_tokens, "effort": think_effort},
        "untrained_direct": {"rate": unt["repair_rate"], "valid": unt["valid"], "tokens": unt["mean_out_tokens"]},
        "vs_reasoning": {k: float(vs_r[k]) for k in ("diff", "lo", "hi", "n_faulty")},
        "vs_untrained": {k: float(vs_u[k]) for k in ("diff", "lo", "hi", "n_faulty")},
    }
    T.update(
        rl_id=pct(rl["repair_rate"]), rl_id_ci=ci(rl["repair_lo"], rl["repair_hi"]), rl_tokens=f"{rl['mean_out_tokens']:.0f}",
        think_id=pct(think["repair_rate"]), think_id_ci=ci(think["repair_lo"], think["repair_hi"]),
        think_tokens=thousands(think["mean_out_tokens"]), think_cap=pct(cap_share, 0), think_cap_tokens=f"{think_cap_tokens:,}", think_finished=pct(think_finished),
        think_effort=f"{think_effort}", rl_temp=f"{rl_settings['temperature']}", rl_max_tokens=f"{rl_settings['max_tokens']}",
        rl_effort=f"{rl_settings['effort']}",
        untrained_id=pct(unt["repair_rate"]), untrained_valid=pct(unt["valid"], 0),
        vs_think=spts(vs_r["diff"]), vs_think_ci=sci(vs_r["lo"], vs_r["hi"]), vs_think_n=f"{int(float(vs_r['n_faulty']))}",
        vs_unt=spts(vs_u["diff"]), vs_unt_ci=sci(vs_u["lo"], vs_u["hi"]),
        n_test=f"{int(rl['n'])}", n_faulty_id=f"{int(rl['n_faulty'])}",
        token_ratio=f"{think['mean_out_tokens'] / rl['mean_out_tokens']:.0f}",
    )

    # --- six test splits ------------------------------------------------------------------------
    rows = []
    for split, short, long in SPLITS:
        a, u, o, s = M["scale_rl"][split], M["untrained"][split], M["baseline_outlier"][split], M["scale_sft"][split]
        rows.append({"split": split, "short": short, "label": long, "n": a["n"], "n_faulty": a["n_faulty"],
                     "trained": a["repair_rate"], "trained_lo": a["repair_lo"], "trained_hi": a["repair_hi"],
                     "untrained": u["repair_rate"], "untrained_lo": u["repair_lo"], "untrained_hi": u["repair_hi"],
                     "outlier": o["repair_rate"], "clean_ok": a["clean_ok"], "sft": s["repair_rate"]})
    D["splits"] = rows
    by = {r["split"]: r for r in rows}
    for split, key in (("test_larger", "larger"), ("test_families", "families"), ("test_faults", "faults"),
                       ("test_trained", "trained"), ("test_external", "external")):
        T[f"rl_{key}"] = pct(by[split]["trained"])
        T[f"rl_{key}_ci"] = ci(by[split]["trained_lo"], by[split]["trained_hi"])
        T[f"rl_{key}_0"] = pct(by[split]["trained"], 0)
    T["rl_id_0"] = pct(by["test_id"]["trained"], 0)
    T["faults_0"] = pct(by["test_faults"]["trained"], 0)
    T["external_clean_bad"] = pct(1 - by["test_external"]["clean_ok"], 0)
    outl = [by[s_]["outlier"] for s_ in ("test_families", "test_larger")]  # the README's "these splits"
    T["outlier_id"] = pct(by["test_id"]["outlier"])
    T["outlier_range"] = f"{100 * min(outl):.0f}-{100 * max(outl):.0f}%"
    T["table_splits"] = "\n".join(
        f"<tr><th scope=row>{esc(r['label'])}</th><td>{pct(r['untrained'])}</td>"
        f"<td class=strong>{pct(r['trained'])} <span class=ci>{ci(r['trained_lo'], r['trained_hi'])}</span></td>"
        f"<td>{pct(r['outlier'])}</td><td>{pct(r['clean_ok'], 0)}</td></tr>" for r in rows)

    # --- progression ("How it got there") -------------------------------------------------------
    cfg = {name: json.load(open(p("runs", name, "config.json"))) for name in ("sft_main", "sft_mixed", "sft_scale")}

    def n_nets(c):
        n = c["steps"] * c["batch_size"]
        return (n, 0) if not c.get("mix") or "repair_trained" not in c["mix"] else (n // 2, n // 2)

    prog_arms = [
        ("sft", "First model, supervised", n_nets(cfg["sft_main"])),
        ("sft_rl", "First model + RL", None),
        ("sft_mixed", "Mixed sources, supervised", n_nets(cfg["sft_mixed"])),
        ("scale_sft", "Scale-up, supervised", n_nets(cfg["sft_scale"])),
        ("scale_rl", "Scale-up + RL (final)", None),
    ]
    prog = []
    trs = []
    for arm, label, nets in prog_arms:
        vals = {s: (M[arm].get(s) or {}).get("repair_rate") for s, _, _ in SPLITS}
        prog.append({"arm": arm, "label": label, "nets": nets, "rates": vals})
        if nets is None:
            ncell = "same"
        elif nets[1] == 0:
            ncell = f"{nets[0]:,} hand-built"
        else:
            ncell = f"{nets[0]:,} + {nets[1]:,}"
        cells = "".join(f"<td>{pct(v) if v is not None else '&mdash;'}</td>" for v in vals.values())
        cls = " class=final" if arm == "scale_rl" else ""
        trs.append(f"<tr{cls}><th scope=row>{esc(label)}</th><td class=muted>{ncell}</td>{cells}</tr>")
    D["progression"] = prog
    T["table_progress"] = "\n".join(trs)
    # reader-facing takeaway of the same progression (no internal model names)
    T["table_takeaway"] = "\n".join([
        f"<tr><th scope=row>Three times the supervised training data</th><td>In-distribution repair</td><td>{pct(M['sft_mixed']['test_id']['repair_rate'], 0)} &rarr; {pct(M['scale_sft']['test_id']['repair_rate'], 0)}</td></tr>",
        f"<tr><th scope=row>Gradient-made networks added to training</th><td>Repair on gradient-made networks</td><td>{pct(M['sft_rl']['test_trained']['repair_rate'], 0)} &rarr; {pct(M['scale_rl']['test_trained']['repair_rate'], 0)}</td></tr>",
        f"<tr><th scope=row>RL on top of supervised training</th><td>Repair, six test sets pooled; healthy networks left alone (larger networks)</td><td>about +{{rl_gain_pooled_0}} points; {pct(M['scale_sft']['test_larger']['clean_ok'], 0)} &rarr; {pct(M['scale_rl']['test_larger']['clean_ok'], 0)}</td></tr>",
    ])
    T["first_sft_id"] = pct(M["sft"]["test_id"]["repair_rate"], 0)
    T["mixed_id_0"] = pct(M["sft_mixed"]["test_id"]["repair_rate"], 0)
    T["scale_sft_id_0"] = pct(M["scale_sft"]["test_id"]["repair_rate"], 0)
    T["first_trained_0"] = pct(M["sft_rl"]["test_trained"]["repair_rate"], 0)

    # RL minus supervised, paired, recomputed from the rollout logs with the repository's own bootstrap
    def succ(arm, split):  # in log order, which is the pairing order the write-up's intervals used
        return {r["seed"]: r["success"] for r in records(p("runs/final", arm, f"{split}.jsonl")) if not r["clean"]}

    pooled_a, pooled_b, per = [], [], {}
    for split, _, _ in SPLITS:
        a, b = succ("scale_rl", split), succ("scale_sft", split)
        seeds = [s for s in a if s in b]
        per[split] = paired_diff_ci([a[s] for s in seeds], [b[s] for s in seeds])
        pooled_a += [a[s] for s in seeds]
        pooled_b += [b[s] for s in seeds]
    pooled = paired_diff_ci(pooled_a, pooled_b)
    src = "recomputed from runs/final logs (log-order pairing)"
    if os.path.exists(p("results/table_rl_vs_sft.csv")):  # the write-up's own table, when present, wins
        rows_ = [r for r in read_csv("table_rl_vs_sft.csv") if r["rl_arm"] == "scale_rl" and r["sft_arm"] == "scale_sft"]
        got = {r["split"]: tuple(snap(r[k], r["n_faulty"]) for k in ("diff", "lo", "hi")) for r in rows_}
        for k in per:
            assert abs(got[k][0] - per[k][0]) < 1e-9, ("RL-minus-SFT point estimate disagrees with the logs", k)
            per[k] = got[k]
        assert abs(got["pooled"][0] - pooled[0]) < 1e-9
        pooled = got["pooled"]
        src = "results/table_rl_vs_sft.csv (point estimates re-checked against the logs)"
    CHECKS.append(f"RL minus supervised: {src}")
    D["rl_minus_sft"] = {"per_split": {k: list(v) for k, v in per.items()}, "pooled": list(pooled)}
    T["rl_gain_id"], T["rl_gain_id_ci"] = spts(per["test_id"][0]), sci(*per["test_id"][1:])
    T["rl_gain_pooled"], T["rl_gain_pooled_ci"] = spts(pooled[0]), sci(*pooled[1:])
    T["rl_gain_pooled_0"] = f"{100 * pooled[0]:.0f}"
    T["clean_larger_sft"] = pct(M["scale_sft"]["test_larger"]["clean_ok"], 0)
    T["clean_larger_rl"] = pct(M["scale_rl"]["test_larger"]["clean_ok"], 0)

    # --- training curves on Tinker ----------------------------------------------------------------
    sft_val = defaultdict(dict)
    for r in read_jsonl(p("runs/sft_scale/metrics.jsonl")):
        if r.get("kind") == "val":
            sft_val[r["step"]][r["val_set"].split("@")[1]] = r["repair_rate"]
    sft_pts = [{"step": s, "val": v["val"], "val_trained": v["val_trained"]} for s, v in sorted(sft_val.items())]
    sft_steps = cfg["sft_scale"]["steps"]
    rl_cfg = json.load(open(p("runs/rl_scale/driver_config.json")))
    chunks = read_jsonl(p("runs/rl_scale/chunks.jsonl"))
    rl_pts = [{"chunk": c["chunk"], "step": sft_steps + c["steps_done"], "val": c["val"]["splits"]["val"]["repair_rate"],
               "val_trained": c["val"]["splits"]["val_trained"]["repair_rate"], "score": c["val"]["mean_repair"]} for c in chunks]
    best = max(rl_pts, key=lambda c: c["score"])
    assert chunks[-1]["best"]["chunk"] == best["chunk"]
    sft_end = sft_pts[-1]
    sft_score = (sft_end["val"] + sft_end["val_trained"]) / 2
    D["training"] = {"sft": sft_pts, "rl": rl_pts, "selected_chunk": best["chunk"], "sft_steps": sft_steps,
                     "sft_score": sft_score, "rl_steps": rl_cfg["chunks"] * rl_cfg["chunk_steps"],
                     "n_val": 300, "n_val_trained": 200}
    T.update(
        rank=f"{cfg['sft_scale']['rank']}", sft_steps=f"{sft_steps}", batch=f"{cfg['sft_scale']['batch_size']}",
        sft_nets=f"{sft_steps * cfg['sft_scale']['batch_size']:,}", sft_lr=f"{cfg['sft_scale']['lr']:.0e}".replace("e-04", "e-4"),
        sft_lr_end=f"{cfg['sft_scale']['lr'] * (1 - cfg['sft_scale']['decay']):.0e}".replace("e-05", "e-5"),
        rl_steps=f"{rl_cfg['chunks'] * rl_cfg['chunk_steps']}", rl_chunks=f"{rl_cfg['chunks']}", rl_chunk_steps=f"{rl_cfg['chunk_steps']}",
        rl_group=f"{rl_cfg['group_size']}", rl_groups=f"{rl_cfg['groups_per_batch']}", rl_lr=f"{rl_cfg['lr']:g}".replace("e-05", "e-5"),
        sel_chunk=f"{best['chunk']}", sel_score=f"{best['score']:.3f}", sft_score=f"{sft_score:.3f}",
        sft_val_end=pct(sft_end["val"]), sft_valt_end=pct(sft_end["val_trained"]), sft_val_start=pct(sft_pts[0]["val"], 0),
        last_chunk=f"{rl_pts[-1]['chunk']}",
    )

    # --- cost -----------------------------------------------------------------------------------
    cost = json.load(open(p("runs/cost_tally.json")))
    by_phase = defaultdict(float)
    for r in cost["runs"]:
        by_phase[(r.get("phase"), r["kind"])] += r["usd"]
    D["cost"] = {"final_model": cost["scale_up"]["total_usd"], "grand_total": cost["grand_total_usd"],
                 "final_sft": by_phase[("scale_up", "sft_train")], "final_rl": by_phase[("scale_up", "rl_train")],
                 "final_eval": by_phase[("scale_up", "eval")]}
    T.update(cost_final=usd0(cost["scale_up"]["total_usd"]), cost_total=usd0(cost["grand_total_usd"]),
             cost_sft=usd2(by_phase[("scale_up", "sft_train")]), cost_rl=usd2(by_phase[("scale_up", "rl_train")]),
             cost_eval=usd2(by_phase[("scale_up", "eval")]))

    # --- comparison: tokens vs accuracy -----------------------------------------------------------
    def mean_reasoning(arm):
        rs = records(p("runs/final", arm, "test_id.jsonl"))
        return float(np.mean([r.get("reasoning_tokens") or 0 for r in rs]))

    comp = [
        {"key": "think", "label": "Untrained Inkling 2 Small, free to reason", "rate": think["repair_rate"], "lo": think["repair_lo"], "hi": think["repair_hi"], "n": think["n_faulty"], "tokens": think["mean_out_tokens"], "open": True},
        {"key": "trained", "label": "Trained Inkling 2 Small, no reasoning", "rate": rl["repair_rate"], "lo": rl["repair_lo"], "hi": rl["repair_hi"], "n": rl["n_faulty"], "tokens": rl["mean_out_tokens"], "open": True},
        {"key": "untrained", "label": "Untrained Inkling 2 Small, no reasoning", "rate": unt["repair_rate"], "lo": unt["repair_lo"], "hi": unt["repair_hi"], "n": unt["n_faulty"], "tokens": unt["mean_out_tokens"], "open": True},
    ]
    # Other models off the shelf, each on the first 100 test networks (results/table_offshelf.csv, written by
    # python -m weightgym.offshelf). One point per model and regime.
    from weightgym.offshelf import ADDED as OFF_ADDED, MODELS as OFF_MODELS, NOT_RUN as OFF_NOT_RUN, tokens_cell as off_tokens_cell
    off = {(r["arm"], r["regime"]): r for r in read_csv("table_offshelf.csv")}
    for (arm, regime), r in off.items():
        if not arm.startswith(("open_", "frontier_")):
            continue
        comp.append({"key": arm, "label": f"{r['model']}, {'no reasoning' if regime == 'direct' else 'with reasoning'}",
                     "rate": float(r["repair_rate"]), "lo": float(r["repair_lo"]), "hi": float(r["repair_hi"]), "n": float(r["n_faulty"]),
                     "n_total": float(r["n"]), "tokens": float(r["mean_out_tokens"]), "reasoning": float(r["mean_reasoning_tokens"] or 0),
                     "open": r["weights"] == "open", "offshelf": True, "cut": float(r["cut_at_token_cap"])})
    D["comparison"] = comp
    base = {k: M[f"baseline_{k}"]["test_id"]["repair_rate"] for k in ("search", "gradient", "outlier", "random", "noop")}
    D["baselines_id"] = base
    T.update(search_id=pct(base["search"]), gradient_id=pct(base["gradient"]), random_id=pct(base["random"]), noop_id=pct(base["noop"], 0))
    fr = {c["key"]: c for c in comp}
    frontier_rates = [c["rate"] for c in comp if c["key"].startswith(("frontier_sol_", "frontier_astra_"))]
    T["frontier_range"] = f"{100 * min(frontier_rates):.0f}-{100 * max(frontier_rates):.0f}%"
    T["offshelf_n"] = f"{max(int(c['n_total']) for c in comp if c.get('offshelf'))}"
    T["frontier_trained_range"] = f"{100 * min(M[a]['test_trained']['repair_rate'] for a in ('frontier_astra_reasoning', 'frontier_sol_reasoning')):.0f}-{100 * max(M[a]['test_trained']['repair_rate'] for a in ('frontier_astra_reasoning', 'frontier_sol_reasoning')):.0f}%"
    trs = []
    for c in comp:
        if c.get("offshelf"):
            continue
        n = f"{int(c['n'])} faulty networks"
        tok = f"~{c['tokens']:,.0f}" + (f" ({c['reasoning']:.0f} reasoning)" if c.get("reasoning") else "")
        cell = pct(c["rate"]) + (f" <span class=ci>{ci(c['lo'], c['hi'])}</span>" if not c.get("partial") else "")
        cls = " class=final" if c["key"] == "trained" else ""
        trs.append(f"<tr{cls}><th scope=row>{esc(c['label'])}</th><td>{cell}</td><td>{tok}</td><td class=muted>{n}</td></tr>")
    for k, label in (("search", "Brute-force search over single sign flips and swaps, using forward passes"),
                     ("gradient", "Gradient descent on half of the inputs (ignores the edit budget)"),
                     ("outlier", "Statistics-only heuristic (most unusual row, column or scalar)"),
                     ("random", "Random valid edit"), ("noop", "Never edit")):
        trs.append(f"<tr><th scope=row>{esc(label)}</th><td>{pct(base[k])}</td><td>&mdash;</td><td class=muted>{int(rl['n_faulty'])} faulty networks</td></tr>")
    T["table_compare"] = "\n".join(trs)

    def off_cells(r, not_run="", bracket=False):
        if r is None:
            return (f"<td class=muted>{esc(not_run)}</td>" if not_run else "<td>&mdash;</td>") + "<td>&mdash;</td><td>&mdash;</td>"
        n = "" if int(r["n"]) == int(T["offshelf_n"]) else f" <span class=ci>n={int(r['n'])}</span>"
        cut = float(r["cut_at_token_cap"])
        return (f"<td>{pct(float(r['repair_rate']))} <span class=ci>{ci(float(r['repair_lo']), float(r['repair_hi']))}</span>{n}</td>"
                f"<td{'' if cut >= 0.25 else ' class=muted'}>{pct(cut, 0)}</td><td>{esc(off_tokens_cell(r, bracket))}</td>")

    otr = [f"<tr class=final><th scope=row>Inkling 2 Small, post-trained on Tinker</th><td>open</td>{off_cells(off.get(('scale_rl', 'direct')), bracket=True)}<td class=muted>not used</td><td>&mdash;</td><td>&mdash;</td></tr>",
           f"<tr><th scope=row>Inkling 2 Small, untrained</th><td>open</td>{off_cells(off.get(('untrained', 'direct')), bracket=True)}{off_cells(off.get(('untrained_thinking', 'reasoning')))}</tr>"]
    for m in OFF_MODELS + OFF_ADDED:
        prefix, name, _id, open_w = m
        d_, r_ = off.get((f"{prefix}_direct", "direct")), off.get((f"{prefix}_reasoning", "reasoning"))
        if d_ is None and r_ is None:
            continue
        star = " *" if m in OFF_ADDED else ""
        otr.append(f"<tr><th scope=row>{esc(name)}{star}</th><td>{'open' if open_w else 'closed'}</td>"
                   f"{off_cells(d_, OFF_NOT_RUN.get((prefix, 'direct'), ''), True)}{off_cells(r_, OFF_NOT_RUN.get((prefix, 'reasoning'), ''))}</tr>")
    T["table_offshelf"] = "\n".join(otr)
    D["offshelf_same"] = {k: {"rate": float(off[a]["repair_rate"]), "lo": float(off[a]["repair_lo"]), "hi": float(off[a]["repair_hi"]), "tokens": float(off[a]["mean_out_tokens"])}
                          for k, a in (("trained", ("scale_rl", "direct")), ("untrained", ("untrained", "direct")), ("think", ("untrained_thinking", "reasoning"))) if a in off}

    # Sentences around that table: weightgym.offshelf.reading writes them from the same rows, for the README too.
    from weightgym.offshelf import reading as off_reading
    R = off_reading(list(off.values()))
    T["offshelf_n_once_open"], T["offshelf_best_open_direct"] = R["n_once_open"], R["best_open"]
    assert f"repairs {R['trained']} in about" in R["direct"]
    T["offshelf_direct"] = esc(R["direct"]).replace(f"repairs {R['trained']} in about", f"repairs <b>{R['trained']}</b> in about", 1)
    T["offshelf_stuck"], T["offshelf_reasoning"] = esc(R["stuck"]), esc(R["reasoning"])
    T["offshelf_lead"], T["offshelf_note"] = esc(R["method"]), esc(R["note"].replace("\\*", "*"))
    T["offshelf_tldr_reason"] = ("; " + esc(R["tldr_reason"])) if R["tldr_reason"] else ""
    D["offshelf_reading"] = R

    # --- verifier -------------------------------------------------------------------------------
    ver = read_csv("table_verify.csv")
    vd = defaultdict(dict)
    for r in ver:
        vd[(r["arm"], r["split"])][int(r["k"])] = {k: snap(r[k], r["n_faulty"]) for k in ("fixed", "lo", "hi")} | {"distinct": float(r["mean_distinct_answers"]) if r["mean_distinct_answers"] else None}
    vsplits = [("test_id", "In-distribution"), ("test_trained", "Gradient-made"), ("test_external", "scikit-learn")]
    D["verify"] = {"splits": [s for s, _ in vsplits], "labels": [l for _, l in vsplits],
                   "trained": {s: vd[("scale_rl", s)] for s, _ in vsplits}, "random": {s: vd[("baseline_random", s)] for s, _ in vsplits},
                   "one_answer": {s: M["scale_rl"][s]["repair_rate"] for s, _ in vsplits},
                   "search": {s: M["baseline_search"][s]["repair_rate"] for s, _ in vsplits},
                   "gradient": {s: M["baseline_gradient"][s]["repair_rate"] for s, _ in vsplits}}
    for s, key in (("test_id", "id"), ("test_trained", "trained"), ("test_external", "external")):
        v8 = vd[("scale_rl", s)][8]
        T[f"ver_{key}"], T[f"ver_{key}_ci"] = pct(v8["fixed"]), ci(v8["lo"], v8["hi"])
        T[f"ver_rand_{key}"] = pct(vd[("baseline_random", s)][8]["fixed"])
    dist = [vd[("scale_rl", s)][8]["distinct"] for s, _ in vsplits]
    T["ver_distinct"] = f"{min(dist):.1f} to {max(dist):.1f}"
    vrows = [("One answer, no verifier", {s: pct(M["scale_rl"][s]["repair_rate"]) for s, _ in vsplits}, ""),
             ("Best of 8 verified answers", {s: f"{pct(vd[('scale_rl', s)][8]['fixed'])} <span class=ci>{ci(vd[('scale_rl', s)][8]['lo'], vd[('scale_rl', s)][8]['hi'])}</span>" for s, _ in vsplits}, " class=final"),
             ("8 random valid edits, verified", {s: pct(vd[("baseline_random", s)][8]["fixed"]) for s, _ in vsplits}, ""),
             ("Brute-force search over single sign flips and swaps", {s: pct(M["baseline_search"][s]["repair_rate"]) for s, _ in vsplits}, ""),
             ("Gradient descent on half of the inputs (no edit budget)", {s: pct(M["baseline_gradient"][s]["repair_rate"]) for s, _ in vsplits}, "")]
    T["table_verify"] = "\n".join(f"<tr{cls}><th scope=row>{esc(l)}</th>" + "".join(f"<td>{v[s]}</td>" for s, _ in vsplits) + "</tr>" for l, v, cls in vrows)

    # --- improve loop (RL only) -------------------------------------------------------------------
    imp = {(r["split"], r["proposer"]): {k: float(v) for k, v in r.items() if k not in ("split", "proposer")} for r in read_csv("table_improve.csv")}
    ti = lambda name: imp[("test_imperfect", name)]  # noqa: E731
    D["improve"] = {
        "series": {
            "trained": [{"k": k, **ti(f"rl_proposer x{k}")} for k in (1, 8)],
            "random": [{"k": k, **ti(f"random x{k}")} for k in (1, 8, 32, 128)],
            "untrained": [{"k": k, **ti(f"untrained x{k}")} for k in (1, 8)],
            "seed2": [{"k": k, **ti(f"rl_proposer_seed2 x{k}")} for k in (1, 8)],
        },
        "search": ti("full search (hundreds of forward passes)"),
        "families": {"trained": imp[("test_imperfect_families", "rl_proposer x8")], "random": imp[("test_imperfect_families", "random x8")]},
    }
    T.update(imp_rl8=gap(ti("rl_proposer x8")["gap_closed"]), imp_rl8_ci=ci(ti("rl_proposer x8")["gap_lo"], ti("rl_proposer x8")["gap_hi"]),
             imp_rl1=gap(ti("rl_proposer x1")["gap_closed"]), imp_rand8=gap(ti("random x8")["gap_closed"]),
             imp_rand8_ci=ci(ti("random x8")["gap_lo"], ti("random x8")["gap_hi"]), imp_rand1=gap(ti("random x1")["gap_closed"]),
             imp_unt8=gap(ti("untrained x8")["gap_closed"]), imp_rand32=gap(ti("random x32")["gap_closed"]),
             imp_rand128=gap(ti("random x128")["gap_closed"]), imp_search=gap(ti("full search (hundreds of forward passes)")["gap_closed"]),
             imp_seed2=gap(ti("rl_proposer_seed2 x8")["gap_closed"]), imp_seed2_ci=ci(ti("rl_proposer_seed2 x8")["gap_lo"], ti("rl_proposer_seed2 x8")["gap_hi"]),
             imp_improved=pct(ti("rl_proposer x8")["improved"], 0), imp_none=pct(1 - ti("rl_proposer x8")["accepted"], 0),
             imp_fam=gap(imp[("test_imperfect_families", "rl_proposer x8")]["gap_closed"]),
             imp_fam_rand=gap(imp[("test_imperfect_families", "random x8")]["gap_closed"]),
             imp_n=f"{int(ti('rl_proposer x8')['n'])}",
             imp_rl8_0=pct(ti("rl_proposer x8")["gap_closed"], 0), imp_rand8_0=pct(ti("random x8")["gap_closed"], 0),
             imp_unt8_0=pct(ti("untrained x8")["gap_closed"], 0))
    irows = [("Untrained Inkling 2 Small", "untrained", ""), ("Random valid edits", "random", ""), ("RL-trained Inkling 2 Small", "rl_proposer", " class=final"),
             ("RL-trained, second, shorter run", "rl_proposer_seed2", "")]
    trs = []
    for label, key, cls in irows:
        a, b = ti(f"{key} x1"), ti(f"{key} x8")
        trs.append(f"<tr{cls}><th scope=row>{label}</th><td>{gap(a['gap_closed'])}</td><td>{gap(b['gap_closed'])} <span class=ci>{ci(b['gap_lo'], b['gap_hi'])}</span></td><td>{pct(b['improved'], 0)}</td></tr>")
    trs.append(f"<tr><th scope=row>Random edits, 32 / 128 proposals</th><td></td><td>{gap(ti('random x32')['gap_closed'])} / {gap(ti('random x128')['gap_closed'])}</td><td>{pct(ti('random x32')['improved'], 0)} / {pct(ti('random x128')['improved'], 0)}</td></tr>")
    s_ = ti("full search (hundreds of forward passes)")
    trs.append(f"<tr><th scope=row>Exhaustive search over simple edits</th><td></td><td>{gap(s_['gap_closed'])}</td><td>{pct(s_['improved'], 0)}</td></tr>")
    T["table_improve"] = "\n".join(trs)

    # --- real data ------------------------------------------------------------------------------
    rw = {(r["split"], r["proposer"]): {k: float(v) for k, v in r.items() if k not in ("split", "proposer")} for r in read_csv("table_realworld.csv")}
    ru = lambda name: rw[("real_test_unseen", name)]  # noqa: E731
    D["realworld"] = {k: ru(k) for k in ("untrained x1", "untrained x8", "random x1", "random x8", "formula_rl_proposer x1", "formula_rl_proposer x8",
                                         "real_rl_proposer x1", "real_rl_proposer x8", "random x32", "random x128", "gradient on select (300 Adam steps)")}
    T.update(real_rl1=gap(ru("real_rl_proposer x1")["gap_closed"]), real_rand1=gap(ru("random x1")["gap_closed"]),
             real_rl8=gap(ru("real_rl_proposer x8")["gap_closed"]), real_rl8_ci=ci(ru("real_rl_proposer x8")["gap_lo"], ru("real_rl_proposer x8")["gap_hi"]),
             real_rand8=gap(ru("random x8")["gap_closed"]), real_rand8_ci=ci(ru("random x8")["gap_lo"], ru("random x8")["gap_hi"]),
             real_grad=gap(ru("gradient on select (300 Adam steps)")["gap_closed"]), real_n=f"{int(ru('random x8')['n'])}")
    ds = json.load(open(p("data/realworld_datasets.json")))
    D["realworld_datasets"] = ds if isinstance(ds, (list, dict)) else None

    ih = read_csv("table_imagehead.csv")
    form = [r for r in ih if r["proposer"] == "formula_rl_proposer"]
    n_models = len(form)
    unchanged = sum(float(r["change_points"]) == 0 for r in form)
    exceeded = sum(r["exceeds_claimed"] == "True" and float(r["test_after"]) > float(r["test_before"]) and float(r["test_before"]) <= float(r["claimed"]) for r in form)
    best_ih = max(form, key=lambda r: float(r["change_points"]))
    D["imagehead"] = [{"model": r["model"], "dataset": r["dataset"], "before": float(r["test_before"]), "after": float(r["test_after"]),
                       "change": float(r["change_points"]), "claimed": float(r["claimed"])} for r in form]
    T.update(ih_n=f"{n_models}", ih_unchanged=f"{unchanged}", ih_best_change=f"{float(best_ih['change_points']):+.1f}",
             ih_best_before=f"{float(best_ih['test_before']):.0f}%", ih_n_real=f"{sum(1 for r in ih if r['proposer'] == 'real_rl_proposer' and int(r['valid_proposals']) > 0)}")
    assert exceeded == 0, "an edit lifted a model above its claimed accuracy; the page's wording would be wrong"
    CHECKS.append(f"image heads: {n_models} models, {unchanged} unchanged with the formula-trained proposer, none lifted above its claimed accuracy by an edit")

    # pure RL from the base model: no-op rate on faulty training networks per step (sweep logs)
    collapse = {}
    for lr in ("2e-4", "6e-5", "2e-5"):
        pts = []
        for r in read_jsonl(p("runs/sweep", f"rl_lr{lr}", "metrics.jsonl")):
            if "step" in r and "env/all/noop_on_faulty" in r:
                pts.append({"step": r["step"], "noop": r["env/all/noop_on_faulty"], "repaired": r.get("env/all/repaired"), "valid": r.get("env/all/valid")})
        collapse[lr] = pts
    D["pure_rl"] = collapse
    T["collapse_step"] = f"{min(pt['step'] for pt in collapse['2e-4'] if pt['noop'] >= 0.999)}"

    # --- the replayed example (hero race and repair figure) ---------------------------------------
    _, eps = load_split(p("data/test_id.jsonl"))
    ep = next(e for e in eps if e.seed == HERO_SEED)
    rec = {arm: next(r for r in records(p("runs/final", arm, "test_id.jsonl")) if r["seed"] == HERO_SEED)
           for arm in ("scale_rl", "untrained_thinking", "untrained")}
    scored = {arm: score_answer(ep, rec[arm]["answer"]) for arm in rec}
    for arm in rec:  # rescoring the logged answer must reproduce the logged outcome
        assert scored[arm]["success"] == rec[arm]["success"], arm
        assert scored[arm]["valid"] == rec[arm]["valid"], arm
    assert rec["scale_rl"]["success"] and rec["untrained_thinking"]["success"]
    CHECKS.append(f"seed {HERO_SEED}: re-scored the three logged answers with weightgym.env.score_answer; outcomes match the logs")
    net = ep.net
    D["example"] = {
        "seed": ep.seed, "family": ep.family, "fault": ep.fault, "fault_info": ep.fault_info, "variant": ep.variant,
        "spec": ep.task.spec, "input_desc": ep.task.input_desc, "output_desc": ep.task.output_desc,
        "n_weights": net.n_weights, "widths": list(net.widths),
        "W": [np.asarray(W).round(2).tolist() for W in net.Ws], "b": [np.asarray(b).round(2).tolist() for b in net.bs],
        "acc_clean": ep.acc_clean, "acc_faulty": ep.acc_faulty, "inverse_edit": ep.inverse_edit, "n_inputs": int(len(ep.task.X)),
        "prompt": render_prompt(ep, direct=True),
        "answers": {arm: {"answer": rec[arm]["answer"], "n_out": rec[arm]["n_out"], "success": scored[arm]["success"],
                          "valid": scored[arm]["valid"], "acc_after": scored[arm]["acc_after"], "recovery": scored[arm]["recovery"],
                          "ops": scored[arm]["ops"]} for arm in rec},
        "thinking": rec["untrained_thinking"]["thinking"],
    }
    ex = D["example"]
    T.update(ex_seed=f"{ep.seed}", ex_nw=f"{net.n_weights}", ex_spec=esc(ep.task.spec), ex_acc_faulty=pct(ep.acc_faulty, 0),
             ex_acc_clean=pct(ep.acc_clean, 0), ex_think_tokens=f"{rec['untrained_thinking']['n_out']:,}", ex_rl_tokens=f"{rec['scale_rl']['n_out']}",
             ex_unt_tokens=f"{rec['untrained']['n_out']}", ex_unt_acc=pct(scored["untrained"]["acc_after"], 0),
             ex_think_acc=pct(scored["untrained_thinking"]["acc_after"], 0), ex_rl_acc=pct(scored["scale_rl"]["acc_after"], 0),
             ex_prompt=esc(ex["prompt"]), ex_n_inputs=f"{len(ep.task.X):,}",
             ex_rl_answer=esc(rec["scale_rl"]["answer"]), ex_think_answer=esc(rec["untrained_thinking"]["answer"]),
             ex_unt_answer=esc(rec["untrained"]["answer"]),
             ex_median_think=thousands(np.median([r["n_out"] for r in think_recs if not r["clean"]])))
    T["ex_think_tokens_k"] = f"{rec['untrained_thinking']['n_out'] / 1000:.1f}k"

    # --- token-for-token race: the trained model keeps answering faulty test networks in file order,
    # starting with the hero network, until it has generated as many tokens as the untrained model spent
    # on the hero network alone. Every tile is a logged answer with its logged n_out and success flag.
    rl_all = records(p("runs/final/scale_rl/test_id.jsonl"))
    start = next(i for i, r in enumerate(rl_all) if r["seed"] == HERO_SEED)
    budget = rec["untrained_thinking"]["n_out"]
    tiles, used = [], 0
    for r in [r for r in rl_all[start:] + rl_all[:start] if not r["clean"]]:
        if used + r["n_out"] > budget:
            break
        used += r["n_out"]
        ops = r.get("ops") or []
        op = "[]" if not ops else ops[0]["op"] + " " + ops[0]["tensor"] + (f" +{len(ops) - 1}" if len(ops) > 1 else "")
        tiles.append({"seed": r["seed"], "fault": r["fault"], "family": r["family"], "n_out": r["n_out"], "success": bool(r["success"]),
                      "valid": bool(r["valid"]), "op": op, "answer": r["answer"], "cum": used})
    assert tiles[0]["seed"] == HERO_SEED
    D["race"] = {"budget": budget, "tiles": tiles, "used": used, "n": len(tiles), "repaired": sum(t["success"] for t in tiles)}
    T.update(race_n=f"{len(tiles)}", race_rep=f"{D['race']['repaired']}", race_used=f"{used:,}", race_budget=f"{budget:,}",
             race_miss=f"{len(tiles) - D['race']['repaired']}", race_n_after=f"{len(tiles) - 1}")
    CHECKS.append(f"token race: {len(tiles)} faulty test_id networks answered by scale_rl in {used} tokens (budget {budget}), {D['race']['repaired']} repaired, from the logged n_out and success fields")

    # --- 'nothing to memorise': one real compiled circuit under the gym's own symmetries --------
    from weightgym.nets import add_dead_neuron, permute_hidden, rescale_hidden, sample_task, split_neuron

    dtask, canon = sample_task("arith", np.random.default_rng([0, 11]), 0)
    drng = np.random.default_rng(7)
    y0 = canon.forward(dtask.X)
    steps_ = [("Compiled circuit", "The gadget as built, before any disguise.", canon, ["a", "b"])]
    n1 = permute_hidden(canon, 0, np.array([1, 0]))
    steps_.append(("Permute neurons", "Hidden neurons swap places, with their incoming rows and outgoing columns.", n1, ["b", "a"]))
    c = np.array([1.5, 0.8])
    n2 = rescale_hidden(n1, 0, c)
    steps_.append(("Rescale neurons", f"Each neuron's incoming row and bias are multiplied by c = {c[0]:g} and {c[1]:g}; its outgoing column is divided by the same c.", n2, ["b", "a"]))
    n3 = split_neuron(n2, 0, 0, 0.4)
    steps_.append(("Split a neuron", "One neuron becomes two identical copies that share its outgoing weight 0.4 : 0.6.", n3, ["b", "a", "b2"]))
    n4 = add_dead_neuron(n3, 0, dtask.X, drng)
    steps_.append(("Add a decoy", "A neuron with ordinary-looking weights whose bias keeps it off on every input.", n4, ["b", "a", "b2", "d"]))
    variants = []
    for title, note, net_, ids in steps_:
        exact = float(np.max(np.abs(net_.forward(dtask.X) - y0)))
        shown = net_.copy().round_()
        acc_shown = dtask.accuracy(shown)
        assert exact < 1e-9, (title, exact)
        assert acc_shown == 1.0, (title, acc_shown)
        assert np.array_equal(dtask.correct(shown), dtask.correct(canon))
        from weightgym.env import render_weights
        variants.append({"title": title, "note": note, "ids": ids, "text": render_weights(shown), "W": [np.asarray(W).round(2).tolist() for W in shown.Ws],
                         "b": [np.asarray(b_).round(2).tolist() for b_ in shown.bs], "max_diff": exact, "acc_shown": acc_shown,
                         "n_weights": shown.n_weights})
    D["disguise"] = {"spec": dtask.spec, "n_inputs": int(len(dtask.X)), "variants": variants, "family": "arith"}
    T.update(dis_spec=esc(dtask.spec), dis_n=f"{len(dtask.X):,}", dis_steps=f"{len(variants) - 1}")
    CHECKS.append(f"disguise: {len(variants) - 1} symmetries from weightgym.nets applied to a compiled arith circuit; outputs identical on all {len(dtask.X)} inputs (exact weights), and the 2-decimal weights shown are correct on all inputs")

    # --- the replayed propose-and-verify network ------------------------------------------------
    _, ieps = load_split(p("data/test_imperfect.jsonl"))
    iep = next(e for e in ieps if e.seed == IMPROVE_SEED)
    irecs = sorted([r for r in records(p("runs/final_improve/rl_proposer/test_imperfect.jsonl")) if r["seed"] == IMPROVE_SEED], key=lambda r: r["sample"])[:8]
    base_acc = iep.task.accuracy(iep.net)
    props, best_acc, kept = [], base_acc, None
    for r in irecs:
        if not r["valid"] or r["ops"] is None:
            props.append({"answer": r["answer"], "valid": False, "acc": None, "kept": False})
            continue
        acc = iep.task.accuracy(apply_edit(iep.net, validate_edit(r["ops"], iep.net)))
        assert abs((acc - base_acc) - r["gain"]) < 1e-9, "re-applied proposal does not reproduce the logged gain"
        better = acc > best_acc + 1e-9
        if better:
            best_acc, kept = acc, len(props)
        props.append({"answer": r["answer"], "valid": True, "acc": acc, "best_so_far": better, "n_out": r["n_out"]})
    gap_closed = (best_acc - base_acc) / (1 - base_acc)
    D["improve_example"] = {"seed": IMPROVE_SEED, "family": iep.family, "spec": iep.task.spec, "n_weights": iep.net.n_weights,
                            "acc_before": base_acc, "acc_after": best_acc, "kept_index": kept, "gap_closed": gap_closed, "proposals": props}
    by_seed = defaultdict(list)
    for r in records(p("runs/final_improve/rl_proposer/test_imperfect.jsonl")):
        by_seed[r["seed"]].append(r)
    kept_gaps = [g for g in (max([0.0] + [r["recovery"] for r in sorted(v, key=lambda r: r["sample"])[:8] if r["valid"]]) for v in by_seed.values()) if g > 0]
    T["imp_mean_kept"] = pct(float(np.mean(kept_gaps)), 0)
    CHECKS.append(f"improve seed {IMPROVE_SEED}: re-applied all 8 logged proposals; accuracy changes match the logged gains")
    T.update(imp_ex_before=pct(base_acc, 1), imp_ex_after=pct(best_acc, 1), imp_ex_gap=pct(gap_closed, 0), imp_ex_seed=f"{IMPROVE_SEED}",
             imp_ex_nw=f"{iep.net.n_weights}", imp_ex_spec=esc(iep.task.spec),
             imp_ex_rejected=f"{sum(1 for x in props if x['valid'] and x['acc'] <= base_acc + 1e-9)}")

    # --- code excerpts, copied from the source at build time --------------------------------------
    D["code"] = {"sft": excerpt("weightgym/tinker_train.py", [
        ("tc = await service.create_lora_training_client_async", 1),
        ("eps = [train_example(a, step * a.batch_size + i", 4),
        ("sw = await (await tc.save_weights_for_sampler_async", 1),
    ]), "rl": excerpt("weightgym/tinker_train.py", [
        ("from tinker_cookbook.rl import train", 1),
        ("cfg = train.Config(", 1),
        ("learning_rate=a.lr,", 6),
        ("remove_constant_reward_groups=True,", 1),
        ("asyncio.run(train.main(cfg))", 1),
    ])}
    T["code_sft"] = render_code(D["code"]["sft"])
    T["code_rl"] = render_code(D["code"]["rl"])

    # --- reproduce commands, copied from the README ----------------------------------------------
    m = re.search(r"## Reproduce\n\n```bash\n(.*?)```", open(p("README.md")).read(), re.S)
    assert m, "README has no Reproduce block"
    T["reproduce"] = esc(m.group(1).rstrip())
    T["repo_link"] = f' <a href="{esc(REPO_URL)}">{esc(REPO_URL)}</a>' if REPO_URL else ""
    T["table_takeaway"] = T["table_takeaway"].replace("{rl_gain_pooled_0}", T["rl_gain_pooled_0"])
    from weightgym.env import REPAIR_THRESHOLD
    T["repair_threshold"] = f"{REPAIR_THRESHOLD}"

    # --- Qwen slot ------------------------------------------------------------------------------
    qpath = os.environ.get("WG_QWEN_JSON", os.path.join(SITE, "qwen.json"))  # env override is for testing the slot only
    D["qwen"] = json.load(open(qpath)) if os.path.exists(qpath) else None
    if D["qwen"] is not None:
        q = D["qwen"]
        assert isinstance(q.get("rows"), list) and isinstance(q.get("summary"), str), "qwen.json: need rows (list) and summary (str)"
        for r in q["rows"]:
            for k in ("label", "math500", "lo", "hi", "n"):
                assert k in r, f"qwen.json row missing {k}"
        CHECKS.append("qwen.json present and well-formed; Qwen section shown")
    T["limits_llm"] = ("On the real LLM, random edits to the same table did as well as the proposers; the edit that works is in one layer "
                       "of one model and removes one known failure rather than adding ability."
                       if D["qwen"] is not None else "There is no evidence here on real LLMs.")

    return D, T


# ------------------------------------------------------------------------------------------------
# Code excerpts
# ------------------------------------------------------------------------------------------------


def excerpt(path: str, anchors: list[tuple[str, int]]) -> dict:
    lines = open(p(path)).read().split("\n")
    blocks = []
    for needle, n in anchors:
        hits = [i for i, l in enumerate(lines) if needle in l]
        assert len(hits) == 1, f"excerpt anchor not unique in {path}: {needle!r} ({len(hits)})"
        i = hits[0]
        blocks.append({"start": i + 1, "lines": lines[i:i + n]})
    return {"path": path, "blocks": blocks}


def render_code(ex: dict) -> str:
    out = []
    prev_end = None
    indent = min(len(l) - len(l.lstrip()) for b in ex["blocks"] for l in b["lines"] if l.strip())
    for b in ex["blocks"]:
        if prev_end is not None and b["start"] != prev_end + 1:
            out.append('<span class="ln"></span><span class="gap">&middot;&middot;&middot;</span>')
        for j, l in enumerate(b["lines"]):
            out.append(f'<span class="ln">{b["start"] + j}</span>{esc(l[indent:])}')
        prev_end = b["start"] + len(b["lines"]) - 1
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------
# Cross-checks against the write-up
# ------------------------------------------------------------------------------------------------


def check_readme(D: dict, T: dict) -> None:
    readme = open(p("README.md")).read()
    s = {r["split"]: r for r in D["splits"]}
    must = [
        f"0.2% to {T['rl_id']}" if T["untrained_id"] == "0.2%" else f"{T['untrained_id']} to {T['rl_id']}",
        f"**{T['rl_id']}** {T['rl_id_ci']}",
        f"{T['think_id']} {T['think_id_ci']}",
        f"{T['vs_think']} points {T['vs_think_ci']}",
        f"~{T['think_tokens']}", f"~{T['rl_tokens']}",
        f"**{T['vs_unt']} points** {T['vs_unt_ci']}",
        f"same {T['vs_think_n']} faulty networks",
        f"out of its\n20,000-token allowance on {T['think_cap']}" if T["think_cap_tokens"] == "20,000" else T["think_cap"],
        f"repairs {T['think_finished']} of those it finishes", f"repairs {T['think_finished']} of the networks it finishes",
        f"**{T['rl_larger']}** {T['rl_larger_ci']}", f"**{T['rl_families']}** {T['rl_families_ci']}",
        f"**{T['rl_faults']}** {T['rl_faults_ci']}", f"**{T['rl_trained']}** {T['rl_trained_ci']}",
        f"**{T['rl_external']}** {T['rl_external_ci']}",
        f"| {T['search_id']} |", f"| {T['gradient_id']} |", f"| {T['outlier_id']} |", f"| {T['random_id']} / {T['noop_id']} |",
        f"**{T['ver_id']}** {T['ver_id_ci']}", f"**{T['ver_trained']}** {T['ver_trained_ci']}", f"**{T['ver_external']}** {T['ver_external_ci']}",
        f"| {T['ver_rand_id']} | {T['ver_rand_trained']} | {T['ver_rand_external']} |",
        f"**{T['imp_rl8']}** {T['imp_rl8_ci']}", f"{T['imp_rand8']} {T['imp_rand8_ci']}", f"| {T['imp_unt8']} |",
        f"{T['imp_seed2']} {T['imp_seed2_ci']}", f"{T['imp_rand32']} / {T['imp_rand128']}", f"| {T['imp_search']} |",
        f"{T['imp_fam']} [", f"**{T['real_rl1']}**", f"{T['real_rl8']} {T['real_rl8_ci']}", f"{T['real_rand8']} {T['real_rand8_ci']}",
        f"| {T['real_grad']} |",
        f"{T['rl_gain_id']} points in-distribution {T['rl_gain_id_ci']}", f"{T['rl_gain_pooled']} {T['rl_gain_pooled_ci']}",
        f"from {T['clean_larger_sft']} to {T['clean_larger_rl']} on larger networks",
        f"about {T['cost_final']} of Tinker usage", f"supervised stage {T['cost_sft']}, RL {T['cost_rl']}", f"about {T['cost_total']} for the whole project",
        f"rank-{T['rank']} LoRA", f"{T['sft_steps']} steps x {T['batch']} networks ({T['sft_nets']}", f"{T['rl_steps']} steps x {T['rl_groups']} networks",
        f"group of {T['rl_group']} samples", f"{T['sft_score']} after the supervised stage, {T['sel_score']} for the selected RL checkpoint",
        f"(chunk {T['sel_chunk']} of 0-{T['last_chunk']})", f"{T['sft_val_end']}\n  on hand-built networks and {T['sft_valt_end']}",
        f"{T['ih_n']} public image classifiers", f"{T['ih_unchanged']} of {T['ih_n']} were unchanged",
        f"heuristic repairs {T['outlier_range']} on these splits",
    ]
    # progression table rows
    for r in D["progression"]:
        must.append(" | ".join(pct(v) if v is not None else "—" for v in r["rates"].values()).replace("**", ""))
    missing = [m for m in must if m.replace("**", "") not in readme.replace("**", "")]
    if missing:
        raise SystemExit("README.md and the data disagree (or the README words it differently); not found in README:\n  " + "\n  ".join(repr(m) for m in missing))
    CHECKS.append(f"README.md contains {len(must)} figures exactly as computed from results/ and runs/")


# ------------------------------------------------------------------------------------------------
# Page rendering
# ------------------------------------------------------------------------------------------------

FIELD = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def lint_template(tpl: str) -> None:
    """No typed-in results: the template may not contain a percentage, a dollar amount or a decimal
    outside template fields. Code, scripts and styles are excluded."""
    body = re.sub(r"<(script|style|pre)[\s\S]*?</\1>", "", tpl)
    body = FIELD.sub("", body)
    body = re.sub(r"<[^>]+>", " ", body)
    # conventions and names, not results: model names, arXiv ids, the confidence level of intervals
    body = re.sub(r"GPT-6\.1|arXiv \d{4}\.\d{5}|95% (bootstrap )?intervals?", " ", body)
    bad = re.findall(r"[-+]?\d[\d,]*\.?\d*\s?%|\$\s?\d|\b\d+\.\d+\b", body)
    if bad:
        raise SystemExit(f"template contains typed-in numbers (use a template field): {sorted(set(bad))}")


def render(D: dict, T: dict) -> str:
    src = os.path.join(SITE, "src")
    tpl = open(os.path.join(src, "template.html")).read()
    lint_template(tpl)
    used = set(FIELD.findall(tpl))
    unknown = used - set(T)
    if unknown:
        raise SystemExit(f"template uses unknown fields: {sorted(unknown)}")
    page = FIELD.sub(lambda m: T[m.group(1)], tpl)
    css = open(os.path.join(src, "style.css")).read()
    js = open(os.path.join(src, "app.js")).read()
    data = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("/*STYLE*/", css).replace("/*DATA*/", data).replace("/*APP*/", js)
    page = page.replace("{{byline}}", esc(D["byline"]))
    return page


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (o != o):
        return None
    return o


def main() -> None:
    D, T = build()
    D = jsonable(D)
    T["byline"] = esc(BYLINE)
    T["byline_url"] = esc(BYLINE_URL)
    T["repo_url"] = esc(REPO_URL)
    check_readme(D, T)
    D["fields"] = T_public(T)
    with open(os.path.join(SITE, "data.json"), "w") as f:
        json.dump(D, f, indent=1)
    page = render(D, T)
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(page)
    print("wrote site/data.json and site/index.html")
    for c in CHECKS:
        print("  check:", c)


def T_public(T: dict) -> dict:
    """Display strings that appear in prose, kept in data.json so the rendered page can be checked against it."""
    skip = ("table_", "code_", "ex_prompt", "reproduce")
    return {k: v for k, v in T.items() if not k.startswith(skip)}


if __name__ == "__main__":
    main()
