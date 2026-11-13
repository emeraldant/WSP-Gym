"""Weight edits on a real LLM: Qwen3.5-0.8B on MATH, with the evaluation protocol of Takehi et al. and Beaker.

Runs on a local GPU against a vLLM server (``scripts/qwen_serve.sh``), with the Python environment that has
vLLM, math-verify and datasets installed (not the Tinker environment).

Protocol (from the released code of "Sparks of Weight-Space Programming", exp8_math, and Beaker's post):
the chat template with thinking on; user message = problem + "\\nPlease reason step by step, and put your
final answer within \\boxed{}."; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, at most 32,768
new tokens; the last \\boxed{} answer after the last "</think>" is graded with math-verify.

An edit is a list of operations on MLP matrices, e.g. ``{"layer": 22, "matrix": "down", "neuron": 573,
"op": "scale", "factor": 2.0}`` (multiply one neuron's output column), and is served as a low-rank adapter:
scaling column j of W by k is W + (k - 1) * W[:, j] e_j^T, a rank-1 update, so many edits can be scored
on one server without rebuilding a checkpoint.

    python -m weightgym.qwen_bench prepare
    python -m weightgym.qwen_bench adapter x2 '[{"layer":22,"matrix":"down","neuron":573,"op":"scale","factor":2}]'
    python -m weightgym.qwen_bench eval --arm unedited --set math500 --n 500
    python -m weightgym.qwen_bench eval --arm x2 --adapter x2 --set math500 --n 500
    python -m weightgym.qwen_bench eval --arm wait_ban --ban-wait --set math500 --n 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

HOME = os.path.expanduser(os.environ.get("WG_QWEN_HOME", "~/wg-qwen"))
BASE = os.path.join(HOME, "models", "base")
BENCH = os.path.join(HOME, "bench")          # benchmark and practice problems stay outside the repository
ADAPTERS = os.path.join(HOME, "adapters")
URL = os.environ.get("WG_QWEN_URL", "http://127.0.0.1:8011/v1")
OUT = "runs/qwen"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 1.5}
TOP_K = 20
MAX_TOKENS = 32768
WAIT_TOKENS = (13784, 13428)                 # "Wait" and " Wait" in the Qwen3.5 tokenizer (checked in tests)
MATRICES = {"down": "down_proj", "up": "up_proj", "gate": "gate_proj"}
LORA_RANK = 8                                # every adapter is padded to this rank
N_LAYERS, D_MODEL, D_MLP = 24, 1024, 3584


# ------------------------------------------------------------------------------------------ data
def prepare() -> None:
    """MATH-500 (held out) and a fixed practice set: 128 GSM8K-train + 128 MATH-train problems, as in Beaker."""
    import random

    from datasets import load_dataset

    os.makedirs(BENCH, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(BENCH, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))

    dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                     for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    test_problems = {r["problem"].strip() for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    gsm = [{"id": f"gsm8k_train_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
           for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train"))]
    mt = []
    for i, r in enumerate(load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")):
        gold = last_boxed(r["solution"])
        if gold is not None and r["problem"].strip() not in test_problems:
            mt.append({"id": f"math_train_{i}", "problem": r["problem"], "answer": gold})
    rng = random.Random(0)
    rng.shuffle(gsm)
    rng.shuffle(mt)
    # disjoint practice sets: "train" scores edits inside a loop, "dev" is a second draw for noise estimates
    dump("train", gsm[:128] + mt[:128])
    dump("dev", gsm[128:256] + mt[128:256])


def load_set(name: str, n: int | None = None) -> list[dict]:
    """Rows of a problem set; ``idx`` is the row's position in the whole set (it fixes the sampling seed)."""
    rows = [{**json.loads(line), "idx": i} for i, line in enumerate(open(os.path.join(BENCH, f"{name}.jsonl")))]
    return rows[:n] if n else rows


# ------------------------------------------------------------------------------------------ grading
_BOX = re.compile(r"\\boxed|\\fbox")


def answer_text(text: str) -> str:
    """The part of a response the grader reads: everything after the last '</think>' (all of it if there is none)."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _box_at(text: str, idx: int) -> str | None:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{..} (else the last \\fbox{..}); an unclosed or empty box is no answer.
    Same rules as the grader released with Takehi et al. (exp8_math/code/mathqa.py)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


_PARSED: dict[str, object] = {}


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    import math_verify

    def parsed(s):
        if s not in _PARSED:
            _PARSED[s] = math_verify.parse("\\boxed{" + s + "}", parsing_timeout=5)
        return _PARSED[s]

    g, p = parsed(gold), parsed(pred)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=5))
    except Exception:  # noqa: BLE001 - an unparseable pair is a wrong answer
        return False


def grade(row: dict, text: str, finish: str, correct: bool | None = None) -> dict:
    pred = last_boxed(answer_text(text))
    if correct is None:
        correct = is_correct(pred, row["answer"])
    return {"pred": pred if pred is None or len(pred) <= 200 else pred[:200] + "...", "correct": bool(correct),
            "cut": finish == "length", "waits": len(re.findall(r"\bWait\b", text)), "chars": len(text)}


class Grader:
    """math-verify in worker processes with a hard time limit.

    A rare malformed answer makes the symbolic comparison run without end, and math-verify's own signal-based
    timeout does not stop it. Run in-process that freezes the whole evaluation (it did, on rented GPUs). Here each
    check runs in a worker process; one that passes ``timeout`` seconds is killed, the answer is counted as wrong
    (as an unparseable answer is), and the record is marked ``grader_timeout``."""

    def __init__(self, workers: int = 4, timeout: float = 30.0):
        self.timeout, self.workers = timeout, workers
        self._free: asyncio.Queue | None = None

    @staticmethod
    def _new():
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    async def correct(self, pred: str | None, gold: str | None) -> tuple[bool, bool]:
        """(correct, timed_out)"""
        if pred is None or gold is None:
            return False, False
        if self._free is None:
            self._free = asyncio.Queue()
            for _ in range(self.workers):
                self._free.put_nowait(self._new())
        ex = await self._free.get()
        try:
            ok = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(ex, is_correct, pred, gold), self.timeout)
            return bool(ok), False
        except Exception:  # noqa: BLE001 - time limit, or a worker that died: count the answer as wrong
            for proc in list(getattr(ex, "_processes", {}).values()):
                proc.kill()
            ex.shutdown(wait=False, cancel_futures=True)
            ex = self._new()
            return False, True
        finally:
            self._free.put_nowait(ex)

    def close(self) -> None:
        while self._free is not None and not self._free.empty():
            self._free.get_nowait().shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------------------------------ edits -> adapters
def validate_edit(ops: list[dict], max_ops: int = 3) -> list[dict]:
    """A Qwen edit: at most ``max_ops`` operations, each on one neuron of one MLP layer.

    {"layer": L, "matrix": "down", "neuron": j, "op": "scale", "factor": k}   output column j of layer L times k
    {"layer": L, "matrix": "up" | "gate", "neuron": j, "op": "scale", "factor": k}   input row j times k
    """
    if not isinstance(ops, list) or len(ops) > max_ops:
        raise ValueError(f"an edit is a list of at most {max_ops} operations")
    out = []
    for o in ops:
        if not isinstance(o, dict) or o.get("op") != "scale" or o.get("matrix") not in MATRICES:
            raise ValueError(f"unsupported operation: {o!r}")
        layer, neuron, factor = o.get("layer"), o.get("neuron"), o.get("factor")
        if not (isinstance(layer, int) and 0 <= layer < N_LAYERS and isinstance(neuron, int) and 0 <= neuron < D_MLP):
            raise ValueError(f"layer or neuron out of range: {o!r}")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not (-16 <= factor <= 16):
            raise ValueError(f"factor must be a number in [-16, 16]: {o!r}")
        out.append({"layer": layer, "matrix": o["matrix"], "neuron": neuron, "op": "scale", "factor": float(factor)})
    return out


def edit_name(ops: list[dict]) -> str:
    import hashlib

    return "e" + hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest()[:12]


_WEIGHTS: dict[str, object] = {}


def weight(layer: int, matrix: str):
    """One MLP matrix of the base model as a float32 torch tensor (cached)."""
    import glob

    import torch
    from safetensors import safe_open

    key = f"model.language_model.layers.{layer}.mlp.{MATRICES[matrix]}.weight"
    if key not in _WEIGHTS:
        with safe_open(glob.glob(os.path.join(BASE, "*.safetensors"))[0], "pt") as f:
            _WEIGHTS[key] = f.get_tensor(key).to(torch.float32)
    return _WEIGHTS[key]


def edit_delta(ops: list[dict]) -> dict[tuple[int, str], list[tuple[object, object]]]:
    """Per edited matrix, the rank-1 terms (b, a) with delta W = sum b a^T. Operations compose in order."""
    import torch

    terms: dict[tuple[int, str], list] = {}
    scale: dict[tuple[int, str, int], float] = {}
    for o in ops:  # two scalings of the same neuron multiply
        k = (o["layer"], o["matrix"], o["neuron"])
        scale[k] = scale.get(k, 1.0) * o["factor"]
    for (layer, matrix, j), f in scale.items():
        W = weight(layer, matrix)
        if matrix == "down":      # column j of a (1024, 3584) matrix
            b, a = (f - 1.0) * W[:, j].clone(), torch.zeros(W.shape[1])
            a[j] = 1.0
        else:                     # row j of a (3584, 1024) matrix
            b, a = torch.zeros(W.shape[0]), (f - 1.0) * W[j, :].clone()
            b[j] = 1.0
        terms.setdefault((layer, matrix), []).append((b, a))
    return terms


def write_adapter(name: str, ops: list[dict]) -> str:
    """Write the edit as a PEFT-format LoRA adapter directory that vLLM can load at runtime."""
    import torch
    from safetensors.torch import save_file

    ops = validate_edit(ops, max_ops=64)
    terms = edit_delta(ops)
    path = os.path.join(ADAPTERS, name)
    os.makedirs(path, exist_ok=True)
    tensors, modules = {}, []
    for (layer, matrix), pairs in terms.items():
        assert len(pairs) <= LORA_RANK, "too many edited neurons in one matrix for the adapter rank"
        mod = f"model.language_model.layers.{layer}.mlp.{MATRICES[matrix]}"
        W = weight(layer, matrix)
        A, B = torch.zeros(LORA_RANK, W.shape[1]), torch.zeros(W.shape[0], LORA_RANK)
        for i, (b, a) in enumerate(pairs):
            B[:, i], A[i, :] = b, a
        tensors[f"base_model.model.{mod}.lora_A.weight"] = A.to(torch.bfloat16)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = B.to(torch.bfloat16)
        modules.append(mod)
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "base_model_name_or_path": BASE, "task_type": "CAUSAL_LM", "r": LORA_RANK,
           "lora_alpha": LORA_RANK, "lora_dropout": 0.0, "bias": "none", "target_modules": modules, "inference_mode": True}
    json.dump(cfg, open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    json.dump(ops, open(os.path.join(path, "edit.json"), "w"), indent=1)
    return path


def write_delta_adapter(name: str, deltas: dict, meta: dict | None = None) -> str:
    """An arbitrary low-rank edit as an adapter: ``deltas[(layer, matrix)] = (B, A)`` with delta W = B @ A
    (numpy arrays, at most LORA_RANK columns in B)."""
    import torch
    from safetensors.torch import save_file

    path = os.path.join(ADAPTERS, name)
    os.makedirs(path, exist_ok=True)
    tensors, modules = {}, []
    for (layer, matrix), (B, A) in deltas.items():
        r = B.shape[1]
        assert r <= LORA_RANK and A.shape[0] == r, "edit rank exceeds the adapter rank"
        mod = f"model.language_model.layers.{layer}.mlp.{MATRICES[matrix]}"
        Bp, Ap = torch.zeros(B.shape[0], LORA_RANK), torch.zeros(LORA_RANK, A.shape[1])
        Bp[:, :r], Ap[:r] = torch.from_numpy(B).float(), torch.from_numpy(A).float()
        tensors[f"base_model.model.{mod}.lora_A.weight"] = Ap.to(torch.bfloat16)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = Bp.to(torch.bfloat16)
        modules.append(mod)
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "base_model_name_or_path": BASE, "task_type": "CAUSAL_LM", "r": LORA_RANK,
           "lora_alpha": LORA_RANK, "lora_dropout": 0.0, "bias": "none", "target_modules": modules, "inference_mode": True}
    json.dump(cfg, open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    json.dump(meta or {}, open(os.path.join(path, "edit.json"), "w"), indent=1)
    return path


def merged_checkpoint(name: str, ops: list[dict]) -> str:
    """The edit written into a full copy of the checkpoint (bfloat16, rounded once): the reference the adapter
    route is checked against, and the form in which an edit would be released."""
    import glob
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    path = os.path.join(HOME, "models", name)
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(BASE):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(BASE, f)):
            shutil.copy(os.path.join(BASE, f), os.path.join(path, f))
    src = glob.glob(os.path.join(BASE, "*.safetensors"))[0]
    sd = load_file(src)
    for (layer, matrix), pairs in edit_delta(validate_edit(ops, max_ops=64)).items():
        key = f"model.language_model.layers.{layer}.mlp.{MATRICES[matrix]}.weight"
        W = sd[key].to(torch.float32)
        for b, a in pairs:
            W = W + torch.outer(b, a)
        sd[key] = W.to(torch.bfloat16)
    save_file(sd, os.path.join(path, os.path.basename(src)), metadata={"format": "pt"})
    return path


# ------------------------------------------------------------------------------------------ server
def _client():
    from openai import AsyncOpenAI

    # a request that gets no answer within the limit fails instead of holding a (possibly rented) GPU
    return AsyncOpenAI(base_url=URL, api_key="local", timeout=float(os.environ.get("WG_REQUEST_TIMEOUT", 5400)), max_retries=0)


async def load_adapter(name: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=120) as h:
        models = (await h.get(URL + "/models")).json()
        if any(m["id"] == name for m in models["data"]):
            return
        r = await h.post(URL.replace("/v1", "") + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": os.path.join(ADAPTERS, name)})
        r.raise_for_status()


async def unload_adapter(name: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=120) as h:
        await h.post(URL.replace("/v1", "") + "/v1/unload_lora_adapter", json={"lora_name": name})


async def generate(client, model: str, problem: str, seed: int, max_tokens: int, ban_wait: bool, sem) -> tuple[str, str, int]:
    kw = dict(SAMPLING)
    if ban_wait:
        kw["logit_bias"] = {str(t): -100 for t in WAIT_TOKENS}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": problem + "\n" + INSTRUCTION}], max_tokens=max_tokens, seed=seed,
                    extra_body={"top_k": TOP_K, "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False}, **kw)
                ch = r.choices[0]
                return ch.message.content or "", ch.finish_reason, r.usage.completion_tokens
            except Exception:  # noqa: BLE001 - connection hiccups on long requests
                if attempt == 3:
                    raise
                await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


async def generate_records(todo: list[tuple[dict, int]], model: str = "base", ban_wait: bool = False, max_tokens: int = MAX_TOKENS,
                           concurrency: int = 256, on_record=None) -> list[dict]:
    """Generate and grade one answer for every (row, sample) pair. The seed of a pair is 1000 * row["idx"] + sample,
    so a pair gets the same seed wherever and in whatever order it is run."""
    client, sem, grader = _client(), asyncio.Semaphore(concurrency), Grader()

    async def one(row, s):
        text, finish, n_tok = await generate(client, model, row["problem"], 1000 * row["idx"] + s, max_tokens, ban_wait, sem)
        ok, timed_out = await grader.correct(last_boxed(answer_text(text)), row["answer"])
        rec = {"id": row["id"], "sample": s, "finish": finish, "tokens": n_tok, **grade(row, text, finish, ok),
               "head": text[:300], "tail": text[-600:]}
        if timed_out:
            rec["grader_timeout"] = True
        return rec

    out = []
    try:
        for fut in asyncio.as_completed([asyncio.ensure_future(one(row, s)) for row, s in todo]):
            rec = await fut
            out.append(rec)
            if on_record:
                on_record(rec)
    finally:
        grader.close()
        await client.close()
    return out


def _header(arm: str, set_name: str, model: str, ban_wait: bool, max_tokens: int, **extra) -> str:
    return json.dumps({"header": True, "arm": arm, "set": set_name, "model": model, "ban_wait": ban_wait, "max_tokens": max_tokens,
                       "sampling": {**SAMPLING, "top_k": TOP_K}, "instruction": INSTRUCTION, "time": time.time(), **extra}) + "\n"


def tag_of(set_name: str, max_tokens: int) -> str:
    return set_name if max_tokens == MAX_TOKENS else f"{set_name}_cap{max_tokens}"


async def evaluate(arm: str, set_name: str, n: int | None, samples: int, model: str = "base", ban_wait: bool = False,
                   max_tokens: int = MAX_TOKENS, concurrency: int = 256, out_dir: str = OUT, quiet: bool = False) -> dict:
    """Score one condition on one problem set on the local server. Resumable: finished (problem, sample) pairs,
    from this machine or merged in from another one, are kept."""
    rows = load_set(set_name, n)
    tag = tag_of(set_name, max_tokens)
    path = os.path.join(out_dir, arm, f"{tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {(r["id"], r["sample"]): r for r in load_records(arm, tag, out_dir)}
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(_header(arm, set_name, model, ban_wait, max_tokens))
    todo = [(row, s) for row in rows for s in range(samples) if (row["id"], s) not in done]
    t0, k = time.time(), [0]
    with open(path, "a") as f:
        def on_record(rec):
            done[(rec["id"], rec["sample"])] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            k[0] += 1
            if not quiet and k[0] % 100 == 0:
                print(f"[{arm}/{tag}] {k[0]}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)

        await generate_records(todo, model, ban_wait, max_tokens, concurrency, on_record)
    return summarize([done[(row["id"], s)] for row in rows for s in range(samples)])


def summarize(recs: list[dict]) -> dict:
    """Accuracy over all samples, with a bootstrap interval over problems."""
    import numpy as np

    by: dict[str, list[int]] = {}
    for r in recs:
        by.setdefault(r["id"], []).append(int(r["correct"]))
    per = np.array([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(0)
    boot = per[rng.integers(0, len(per), size=(5000, len(per)))].mean(1)
    n = len(recs)
    return {"n_problems": len(by), "n_answers": n, "acc": float(per.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "cut_share": sum(r["cut"] for r in recs) / n, "mean_tokens": sum(r["tokens"] for r in recs) / n,
            "tokens": sum(r["tokens"] for r in recs), "waits_per_10k_chars": 1e4 * sum(r["waits"] for r in recs) / max(1, sum(r["chars"] for r in recs))}


def load_records(arm: str, tag: str, out_dir: str = OUT) -> list[dict]:
    """Records of one condition on one set: ``<tag>.jsonl`` (this machine) plus ``<tag>.<source>.jsonl`` (answers
    generated elsewhere under the same protocol, e.g. ``.modal``). One record per (problem, sample); first seen wins."""
    d = os.path.join(out_dir, arm)
    out: dict[tuple, dict] = {}
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".jsonl") and fn.split(".")[0] == tag:
            for line in open(os.path.join(d, fn)):
                r = json.loads(line)
                if not r.get("header"):
                    out.setdefault((r["id"], r["sample"]), r)
    return list(out.values())


def table(out_dir: str = OUT, results: str = "results/table_qwen.csv") -> None:
    import csv

    rows = []
    for arm in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, arm)
        if not os.path.isdir(d):
            continue
        for tag in sorted({f.split(".")[0] for f in os.listdir(d) if f.endswith(".jsonl")}):
            recs = load_records(arm, tag, out_dir)
            if recs:
                s = summarize(recs)
                rows.append({"arm": arm, "set": tag, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
    if rows:
        os.makedirs(os.path.dirname(results), exist_ok=True)
        with open(results, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(f"{r['arm']:28s} {r['set']:18s} acc {100 * r['acc']:5.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}] "
              f"cut {100 * r['cut_share']:4.0f}% mean tokens {r['mean_tokens']:7.0f} Wait/10k chars {r['waits_per_10k_chars']:.0f} n {r['n_answers']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    a_ = sub.add_parser("adapter")
    a_.add_argument("name")
    a_.add_argument("ops")
    m_ = sub.add_parser("merge")
    m_.add_argument("name")
    m_.add_argument("ops")
    e = sub.add_parser("eval")
    e.add_argument("--arm", required=True)
    e.add_argument("--set", default="math500")
    e.add_argument("--n", type=int, default=None)
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--adapter", default=None, help="name of an adapter written by the 'adapter' command")
    e.add_argument("--ban-wait", action="store_true")
    e.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    e.add_argument("--concurrency", type=int, default=256)
    sub.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "adapter":
        print(write_adapter(a.name, json.loads(a.ops)))
    elif a.cmd == "merge":
        print(merged_checkpoint(a.name, json.loads(a.ops)))
    elif a.cmd == "eval":
        async def run():
            if a.adapter:
                await load_adapter(a.adapter)
            return await evaluate(a.arm, a.set, a.n, a.samples, a.adapter or "base", a.ban_wait, a.max_tokens, a.concurrency)

        s = asyncio.run(run())
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    else:
        table()


if __name__ == "__main__":
    main()
