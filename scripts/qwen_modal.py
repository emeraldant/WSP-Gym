"""Run the Qwen3.5-0.8B evaluations of weightgym/qwen_bench.py on Modal GPUs.

Each container starts the same vLLM server as scripts/qwen_serve.sh (same vLLM version, same sampler setting) and
runs the same generation and grading code; only the hardware differs. Results are merged into the local result
files as ``<set>.modal.jsonl`` next to anything generated on the local GPU.

    python scripts/qwen_modal.py repro --start 250 --stop 500        # the three reproduction conditions
    python scripts/qwen_modal.py baseline --cap 4096                 # unedited x4 seeds and neuron 573 x2 on practice sets
    python scripts/qwen_modal.py screen | dev                        # slice edits (weightgym/qwen_slice.py)
    python scripts/qwen_modal.py final <edit_id> ...                 # full MATH-500 protocol for chosen edits
    python scripts/qwen_modal.py ledger

A ledger of container time is kept in runs/qwen/modal_ledger.jsonl; a batch is refused if the ledger plus the
batch's estimate would pass BUDGET_USD.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_REV = "2fc06364715b967f1860aea9cf38778875588b17"
GPU = os.environ.get("WG_MODAL_GPU", "H100")
PRICE_PER_S = {"H100": 0.001097, "A100-80GB": 0.000694, "L40S": 0.000542}[GPU] + 8 * 0.0000131 + 0.0002   # GPU + CPU cores + memory (rough)
BUDGET_USD = float(os.environ.get("WG_MODAL_BUDGET", 26.0))
LEDGER = "runs/qwen/modal_ledger.jsonl"
STARTUP_S = 240          # image pull + server start + idle before scale-down, billed per container (estimate)


def _download():
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3.5-0.8B", revision=MODEL_REV, local_dir="/wg/models/base")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "transformers==5.20.2", "math-verify==0.9.0", "openai==3.50.0", "httpx==0.28.1", "huggingface_hub")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "WG_QWEN_HOME": "/wg",
          "HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_function(_download)
    .add_local_python_source("weightgym")
)
app = modal.App("weightgym-qwen", image=image)


@app.cls(gpu=GPU, cpu=8, memory=32768, timeout=3 * 3600, scaledown_window=30, max_containers=8)
class Qwen:
    @modal.enter()
    def start(self):
        import httpx

        self.t_enter = time.time()
        cmd = ["vllm", "serve", "/wg/models/base", "--served-model-name", "base", "--port", "8011", "--host", "127.0.0.1",
               "--dtype", "bfloat16", "--max-model-len", "40000", "--gpu-memory-utilization", "0.92", "--enable-lora",
               "--max-lora-rank", "8", "--max-loras", "16", "--max-cpu-loras", "64", "--max-num-seqs", "512", "--seed", "0"]
        self.proc = subprocess.Popen(cmd, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("vLLM exited: " + open("/tmp/vllm.log").read()[-3000:])
            try:
                if httpx.get("http://127.0.0.1:8011/v1/models", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.time() - self.t_enter > 900:
                raise RuntimeError("vLLM did not start: " + open("/tmp/vllm.log").read()[-3000:])
            time.sleep(3)
        self.startup = time.time() - self.t_enter

    @modal.method()
    async def run(self, jobs: list[dict]) -> dict:
        """jobs: {"key", "todo": [(row, sample)], "ban_wait", "cap", "edit": None | {"name", "ops"} | {"name", "layer", "B", "A"}}.
        Jobs in one call run concurrently on this container's server."""
        import asyncio

        import numpy as np

        from weightgym import qwen_bench as qb

        t0 = time.time()

        async def one(job):
            model, edit = "base", job.get("edit")
            if edit:
                model = edit["name"]
                if "ops" in edit:
                    qb.write_adapter(model, edit["ops"])
                else:
                    qb.write_delta_adapter(model, {(edit["layer"], "down"): (np.asarray(edit["B"], dtype=np.float32), np.asarray(edit["A"], dtype=np.float32))})
                await qb.load_adapter(model)
            try:
                recs = await qb.generate_records([(r, s) for r, s in job["todo"]], model, job.get("ban_wait", False), job["cap"], 512)
            finally:
                if edit:
                    await qb.unload_adapter(model)
            return {"key": job["key"], "records": recs}

        out = await asyncio.gather(*(one(j) for j in jobs))
        return {"results": out, "seconds": time.time() - t0, "startup": self.startup, "gpu": GPU,
                "tokens": sum(r["tokens"] for o in out for r in o["records"])}

    @modal.exit()
    def stop(self):
        self.proc.terminate()


# ------------------------------------------------------------------------------------------ local driver
def spent() -> float:
    return sum(json.loads(line)["usd"] for line in open(LEDGER)) if os.path.exists(LEDGER) else 0.0


def _log(entry: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_batches(batches: list[list[dict]], sink, label: str, est_tokens: float, tok_per_s: float = 12000.0) -> None:
    """One container per batch. ``sink(key, records)`` stores results as they arrive."""
    batches = [b for b in batches if b]
    if not batches:
        print(f"[{label}] nothing to do")
        return
    est = (est_tokens / tok_per_s + STARTUP_S * len(batches)) * PRICE_PER_S
    if spent() + est > BUDGET_USD:
        raise SystemExit(f"[{label}] refused: spent ${spent():.2f} + estimate ${est:.2f} would pass the ${BUDGET_USD:.0f} budget")
    print(f"[{label}] {len(batches)} containers on {GPU}, estimate ${est:.2f} (spent so far ${spent():.2f})", flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        for res in Qwen().run.map(batches, order_outputs=False, return_exceptions=True):
            if isinstance(res, Exception):
                print(f"[{label}] a container failed: {res!r}"[:600], flush=True)
                _log({"label": label, "t": time.time(), "usd": STARTUP_S * PRICE_PER_S, "failed": True})
                continue
            for o in res["results"]:
                sink(o["key"], o["records"])
            usd = (res["seconds"] + res["startup"] + 60) * PRICE_PER_S
            _log({"label": label, "t": time.time(), "gpu": res["gpu"], "seconds": round(res["seconds"]), "startup": round(res["startup"]),
                  "tokens": res["tokens"], "usd": round(usd, 3)})
            print(f"[{label}] container done: {res['tokens'] / 1e6:.2f}M tokens in {res['seconds']:.0f}s "
                  f"({res['tokens'] / max(1, res['seconds']):.0f} tok/s, startup {res['startup']:.0f}s), ${usd:.2f}", flush=True)
    print(f"[{label}] finished in {time.time() - t0:.0f}s; ledger total ${spent():.2f}", flush=True)


def file_sink(out_dir: str):
    """Append records to <out_dir>/<arm>/<tag>.modal.jsonl; key = (arm, tag)."""
    def sink(key, records):
        arm, tag = key
        path = os.path.join(out_dir, arm, f"{tag}.modal.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sink


def pending(arm: str, set_name: str, rows: list[dict], samples, cap: int, out_dir: str) -> list[tuple[dict, int]]:
    from weightgym import qwen_bench as qb

    done = {(r["id"], r["sample"]) for r in qb.load_records(arm, qb.tag_of(set_name, cap), out_dir)}
    return [(row, s) for row in rows for s in samples if (row["id"], s) not in done]


def split(todo: list, n: int) -> list[list]:
    return [todo[i::n] for i in range(n)] if todo else []


X2 = [{"layer": 22, "matrix": "down", "neuron": 573, "op": "scale", "factor": 2.0}]


def cmd_repro(start: int, stop: int, containers: int) -> None:
    from weightgym import qwen_bench as qb

    rows = qb.load_set("math500")[start:stop]
    conds = [("unedited", None, False), ("n573_x2", {"name": "x2", "ops": X2}, False), ("wait_ban", None, True)]
    batches = [[] for _ in range(containers)]
    total = 0
    for arm, edit, ban in conds:
        todo = pending(arm, "math500", rows, [0], qb.MAX_TOKENS, qb.OUT)
        total += len(todo)
        for i, part in enumerate(split(todo, containers)):
            if part:
                batches[i].append({"key": (arm, "math500"), "todo": part, "ban_wait": ban, "cap": qb.MAX_TOKENS, "edit": edit})
    run_batches(batches, file_sink(qb.OUT), f"repro[{start}:{stop}]", est_tokens=total * 26000, tok_per_s=9000)
    qb.table()


def cmd_baseline(cap: int, seeds: int, sets: list[str], n_train: int, arms: list[str]) -> None:
    """Unedited and neuron 573 x2 on the practice sets at a reduced token limit: the positive control, the noise
    estimate (several seeds of the unedited model) and the keep threshold for the slice benchmark."""
    from weightgym import qwen_bench as qb

    jobs = []
    for arm, edit in (("unedited", None), ("n573_x2", {"name": "x2", "ops": X2})):
        if arm not in arms:
            continue
        for set_name in sets:
            rows = qb.load_set(set_name)[:n_train] if set_name == "train" else qb.load_set(set_name)
            todo = pending(arm, set_name, rows, list(range(seeds)) if arm == "unedited" and set_name == "train" else [0], cap, qb.OUT)
            if todo:
                jobs.append({"key": (arm, qb.tag_of(set_name, cap)), "todo": todo, "ban_wait": False, "cap": cap, "edit": edit})
    run_batches([jobs], file_sink(qb.OUT), f"baseline cap {cap}", est_tokens=sum(len(j["todo"]) for j in jobs) * cap * 0.9)
    qb.table()


def _slice_jobs(edits: dict, set_name: str, rows: list[dict], cap: int) -> list[dict]:
    from weightgym import qwen_bench as qb
    from weightgym import qwen_slice as qs

    items = {it.layer: it for it in qs.load_items()}
    jobs = []
    for eid, (layer, ops) in edits.items():
        todo = pending(f"edit_{eid}", set_name, rows, [0], cap, os.path.join(qs.OUT, "edits"))
        if todo:
            B, A = items[layer].factors(ops)
            jobs.append({"key": (f"edit_{eid}", qb.tag_of(set_name, cap)), "todo": todo, "ban_wait": False, "cap": cap,
                         "edit": {"name": eid, "layer": layer, "B": B.astype("float32"), "A": A.astype("float32")}})
    return jobs


def cmd_screen(containers: int, limit: int | None) -> None:
    """Score distinct valid slice edits on the screen rows, taking them from the end of the sorted list (a local
    worker, ``python -m weightgym.qwen_slice screen``, takes them from the start)."""
    from weightgym import qwen_slice as qs

    cfg = qs._cfg()
    edits = dict(sorted(qs.distinct_edits().items())[::-1])
    jobs = _slice_jobs(edits, "train", qs.screen_rows(), cfg["cap"])[:limit]
    print(f"{len(edits)} distinct valid edits, {len(jobs)} to score here")
    # at most 16 adapters are live on one server: each container gets one wave of up to 12 edits
    waves = [jobs[i:i + 12] for i in range(0, len(jobs), 12)]
    est = sum(len(j["todo"]) for j in jobs) * cfg["cap"] * 0.9
    for i in range(0, len(waves), containers):
        run_batches(waves[i:i + containers], file_sink(os.path.join(qs.OUT, "edits")), f"screen {i // containers}",
                    est_tokens=est * len(waves[i:i + containers]) / max(1, len(waves)), tok_per_s=9000)


def cmd_dev() -> None:
    from weightgym import qwen_bench as qb
    from weightgym import qwen_slice as qs

    cfg = qs._cfg()
    edits = {qs.edit_id(layer, b[0]): (layer, b[0]) for by in qs.kept_edits().values() for layer, b in by.items() if b}
    jobs = _slice_jobs(edits, "dev", qb.load_set("dev"), cfg["cap"])
    print(f"{len(edits)} kept edits, {len(jobs)} still to score on dev")
    waves = [jobs[i:i + 12] for i in range(0, len(jobs), 12)]
    run_batches(waves, file_sink(os.path.join(qs.OUT, "edits")), "dev", est_tokens=sum(len(j["todo"]) for j in jobs) * cfg["cap"] * 0.9, tok_per_s=9000)


def cmd_final(edit_ids: list[str], containers: int) -> None:
    """Full MATH-500 protocol for chosen slice edits."""
    from weightgym import qwen_bench as qb
    from weightgym import qwen_slice as qs

    known = qs.distinct_edits()
    jobs = _slice_jobs({e: known[e] for e in edit_ids}, "math500", qb.load_set("math500"), qb.MAX_TOKENS)
    batches = [[] for _ in range(containers)]
    for j in jobs:
        for i, part in enumerate(split(j["todo"], containers)):
            if part:
                batches[i].append({**j, "todo": part})
    run_batches(batches, file_sink(os.path.join(qs.OUT, "edits")), "final", est_tokens=sum(len(j["todo"]) for j in jobs) * 24000, tok_per_s=9000)
    qb.table(os.path.join(qs.OUT, "edits"), "results/table_qwen_slice_edits.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repro")
    p.add_argument("--start", type=int, default=250)
    p.add_argument("--stop", type=int, default=500)
    p.add_argument("--containers", type=int, default=3)
    p = sub.add_parser("baseline")
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--sets", default="train")
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--arms", default="unedited,n573_x2")
    p = sub.add_parser("screen")
    p.add_argument("--containers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="score at most this many edits here")
    sub.add_parser("dev")
    p = sub.add_parser("final")
    p.add_argument("edit_ids", nargs="+")
    p.add_argument("--containers", type=int, default=3)
    sub.add_parser("ledger")
    a = ap.parse_args()
    if a.cmd == "repro":
        cmd_repro(a.start, a.stop, a.containers)
    elif a.cmd == "baseline":
        cmd_baseline(a.cap, a.seeds, a.sets.split(","), a.n_train, a.arms.split(","))
    elif a.cmd == "screen":
        cmd_screen(a.containers, a.limit)
    elif a.cmd == "dev":
        cmd_dev()
    elif a.cmd == "final":
        cmd_final(a.edit_ids, a.containers)
    else:
        print(f"spent (estimated from container time) ${spent():.2f} of ${BUDGET_USD:.0f}")
