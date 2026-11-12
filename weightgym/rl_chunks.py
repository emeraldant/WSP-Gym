"""RL in chunks, for accounts on Tinker's free checkpoint-storage plan (100 GB).

An Inkling 2 Small training state is ~50 GB and sampler weights ~17 GB, so the usual pattern of
saving state + sampler every N steps cannot be used: the second save would cross the limit and stop
the session. This driver runs the tinker-cookbook RL loop (``weightgym.tinker_train rl``) in short
chunks instead. Each chunk starts from the previous chunk's final state, saves exactly one final
state + sampler, and the driver keeps on Tinker only:

    * the newest state (to start the next chunk), and
    * the sampler weights with the best validation repair rate so far (checkpoint selection).

The previous state is deleted while the next chunk's last step is running, so at most one state
exists when the chunk writes its final checkpoint. Each chunk starts with a fresh Adam state.
With ``--keep-all`` nothing is deleted (paid storage).

Validation after every chunk uses the frozen ``val`` split with the pre-registered sampling settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import tinker

from .ckpt import delete_paths
from .env import load_split
from .eval import summarize
from .llm import INKLING, LLMPolicy, evaluate


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


async def _validate(service, sampler_path: str, a, out: str) -> dict:
    _, eps = load_split(a.val_split)
    policy = await LLMPolicy(service, a.model, sampler_path, a.effort, a.max_tokens).start()
    recs = await evaluate(policy, eps[: a.n_val], out, 1, 64, tag=os.path.basename(out), direct=True)
    return summarize(recs)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--load", required=True, help="tinker:// state path to start from (e.g. the SFT final state)")
    ap.add_argument("--init-sampler", required=True, help="sampler path of the starting weights")
    ap.add_argument("--init-val", type=float, required=True, help="validation repair rate of the starting weights")
    ap.add_argument("--model", default=INKLING)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--chunk-steps", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--groups-per-batch", type=int, default=32)
    ap.add_argument("--train-offset", type=int, default=100_000)
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--val-split", default="data/val.jsonl")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--keep-init", action="store_true", help="never delete the starting state/sampler")
    a = ap.parse_args()
    os.makedirs(a.log_path, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.log_path, "driver_config.json"), "w"), indent=1)
    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    hist = os.path.join(a.log_path, "chunks.jsonl")
    state, best = a.load, {"chunk": -1, "val": a.init_val, "sampler": a.init_sampler}
    protected = {a.load, a.init_sampler} if a.keep_init else set()
    start = 0
    if os.path.exists(hist):  # resume the driver after an interruption
        for line in open(hist):
            r = json.loads(line)
            start, state = r["chunk"] + 1, r["state_path"]
            best = r["best"]
    rollouts_per_chunk = a.chunk_steps * a.group_size * a.groups_per_batch
    for c in range(start, a.chunks):
        log = os.path.join(a.log_path, f"chunk{c:02d}")
        os.makedirs(log, exist_ok=True)
        cmd = [sys.executable, "-m", "weightgym.tinker_train", "rl", "--log-path", log, "--model", a.model,
               "--lr", str(a.lr), "--steps", str(a.chunk_steps), "--group-size", str(a.group_size),
               "--groups-per-batch", str(a.groups_per_batch), "--eval-every", "0", "--save-every", "0",
               "--load", state, "--effort", str(a.effort), "--max-tokens", str(a.max_tokens),
               "--train-offset", str(a.train_offset + c * a.chunk_steps * a.groups_per_batch)]
        print(f"[chunk {c}] start from {state}", flush=True)
        proc = subprocess.Popen(cmd, stdout=open(os.path.join(log, "stdout.log"), "a"), stderr=subprocess.STDOUT)
        deleted_prev = False
        threshold = rollouts_per_chunk - a.group_size * a.groups_per_batch + 1  # first rollout of the last step
        while proc.poll() is None:
            await asyncio.sleep(5)
            if not deleted_prev and not a.keep_all and state not in protected and \
                    _count_lines(os.path.join(log, "rollouts.jsonl")) >= threshold:
                await delete_paths([state], rest)  # the running chunk no longer needs its starting state
                deleted_prev = True
        if proc.returncode != 0:
            raise SystemExit(f"chunk {c} failed (exit {proc.returncode}); see {log}/stdout.log. "
                             f"Starting state {'was deleted' if deleted_prev else 'still exists'}: {state}")
        ck = [json.loads(line) for line in open(os.path.join(log, "checkpoints.jsonl"))][-1]
        s = await _validate(service, ck["sampler_path"], a, os.path.join(a.log_path, f"val_chunk{c:02d}.jsonl"))
        improved = s["repair_rate"] > best["val"]
        if improved:
            old = best["sampler"]
            best = {"chunk": c, "val": s["repair_rate"], "sampler": ck["sampler_path"]}
            if not a.keep_all and old not in protected:
                await delete_paths([old], rest)
        elif not a.keep_all:
            await delete_paths([ck["sampler_path"]], rest)
        state = ck["state_path"]
        with open(hist, "a") as f:
            f.write(json.dumps({"chunk": c, "t": time.time(), "steps_done": (c + 1) * a.chunk_steps, "state_path": state,
                                "sampler_path": ck["sampler_path"], "sampler_kept": improved or a.keep_all,
                                "val": s, "best": best}) + "\n")
        print(f"[chunk {c}] val repair {s['repair_rate']:.3f} [{s['repair_lo']:.3f},{s['repair_hi']:.3f}] clean_ok {s['clean_ok']:.2f} "
              f"valid {s['valid']:.2f} noop_f {s['noop_faulty']:.2f} | best so far: chunk {best['chunk']} ({best['val']:.3f})", flush=True)
    print(f"[done] best chunk {best['chunk']} val {best['val']:.3f} sampler {best['sampler']} | last state {state}")


if __name__ == "__main__":
    asyncio.run(main())
