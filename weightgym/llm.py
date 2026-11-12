"""Sampling an LLM policy on gym episodes through Tinker, with full JSONL logging.

Both arms of every comparison (untrained base model, trained LoRA checkpoint) go
through this one code path, so prompt, renderer, thinking effort, temperature and
max_tokens are identical by construction.

The API key is read from the environment by the Tinker SDK (TINKER_API_KEY); it is
never written to disk or logs by this module.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .env import Episode, GymConfig, load_split, make_episode, score_answer
from .eval import breakdown, record, summarize

INKLING = "thinkingmachines/Inkling-2-Small"


def renderer_name_for(model_name: str) -> str:
    """Cookbook-recommended renderer; WEIGHTGYM_RENDERER overrides it (used only for loop debugging on a
    cheap non-Inkling model, e.g. qwen3_disable_thinking)."""
    return os.environ.get("WEIGHTGYM_RENDERER") or model_info.get_recommended_renderer_name(model_name)


def split_content(content) -> tuple[str, str]:
    """Return (final_answer_text, thinking_text) from a parsed assistant message."""
    if isinstance(content, str):
        return content, ""
    text, thinking = [], []
    for part in content or []:
        if part.get("type") == "text":
            text.append(part.get("text", ""))
        elif part.get("type") == "thinking":
            thinking.append(part.get("thinking", ""))
    return "".join(text), "".join(thinking)


class LLMPolicy:
    def __init__(self, service: tinker.ServiceClient, base_model: str, model_path: str | None = None,
                 effort: float | None = None, max_tokens: int = 4096, temperature: float = 1.0):
        self.service, self.base_model, self.model_path = service, base_model, model_path
        self.effort, self.max_tokens, self.temperature = effort, max_tokens, temperature
        self.renderer_name = renderer_name_for(base_model)
        self.renderer = get_renderer(self.renderer_name, get_tokenizer(base_model))
        self.client = None

    async def start(self):
        if self.model_path:
            self.client = await self.service.create_sampling_client_async(model_path=self.model_path)
        else:
            self.client = await self.service.create_sampling_client_async(base_model=self.base_model)
        return self

    def build_prompt(self, text: str):
        msgs = [Message(role="user", content=text)]
        if self.effort is None or not self.base_model.startswith("thinkingmachines/"):
            return self.renderer.build_generation_prompt(msgs)  # only Inkling renderers take an effort
        return self.renderer.build_generation_prompt(msgs, effort=self.effort)

    def settings(self) -> dict:
        return {"base_model": self.base_model, "model_path": self.model_path, "renderer": self.renderer_name,
                "effort": self.effort, "max_tokens": self.max_tokens, "temperature": self.temperature}

    async def sample(self, text: str, n: int = 1) -> tuple[int, list[dict]]:
        prompt = self.build_prompt(text)
        params = tinker.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature,
                                       stop=self.renderer.get_stop_sequences())
        out = await self.client.sample_async(prompt=prompt, num_samples=n, sampling_params=params)
        res = []
        for seq in out.sequences:
            msg, term = self.renderer.parse_response(seq.tokens)
            answer, thinking = split_content(msg.get("content"))
            if str(seq.stop_reason) != "stop" or "stop_sequence" not in str(term).lower():
                # Truncated or unparseable: there is no final answer. (The parser falls back to raw
                # text that still contains the thinking, which must not be mined for an answer.)
                answer, thinking = "", (thinking or answer)
            res.append({"answer": answer, "thinking": thinking, "n_out": len(seq.tokens),
                        "stop_reason": str(seq.stop_reason), "termination": str(term)})
        return prompt.length, res


async def evaluate(policy: LLMPolicy, episodes: list[Episode], out_path: str, samples: int = 1,
                   concurrency: int = 48, tag: str = "", retries: int = 3, direct: bool = False) -> list[dict]:
    """Sample ``samples`` answers per episode, score them, append one JSONL line per rollout."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done: set[tuple[int, int]] = set()
    records: list[dict] = []
    if os.path.exists(out_path):  # resume
        for line in open(out_path):
            r = json.loads(line)
            if "seed" in r:
                done.add((r["seed"], r["sample"]))
                records.append(r)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    f = open(out_path, "a")
    if not records:
        f.write(json.dumps({"header": True, "tag": tag, "settings": policy.settings(), "direct": direct, "time": time.time()}) + "\n")
    t0 = time.time()

    async def one(ep: Episode):
        need = [k for k in range(samples) if (ep.seed, k) not in done]
        if not need:
            return
        text = ep.prompt(direct)
        async with sem:
            for attempt in range(retries):
                try:
                    n_in, outs = await policy.sample(text, len(need))
                    break
                except Exception as e:  # noqa: BLE001 - log and retry transient API errors
                    if attempt == retries - 1:
                        print(f"[warn] seed {ep.seed}: sampling failed: {e!r}")
                        return
                    await asyncio.sleep(2 * 2**attempt)
        async with lock:
            for k, o in zip(need, outs):
                res = ep.score(o["answer"]) if hasattr(ep, "score") else score_answer(ep, o["answer"])
                rec = record(ep, res, sample=k, tag=tag, direct=direct, prompt_hash=ep.prompt_hash(direct), n_in=n_in, n_out=o["n_out"],
                             stop_reason=o["stop_reason"], termination=o["termination"], error=res["error"],
                             ops=res["ops"], answer=o["answer"], thinking=o["thinking"])
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
            f.flush()

    await asyncio.gather(*(one(ep) for ep in episodes))
    f.close()
    new = [r for r in records if (r["seed"], r["sample"]) not in done]
    if new:
        print(f"[{tag}] {len(new)} rollouts in {time.time() - t0:.0f}s; "
              f"tokens in {sum(r['n_in'] for r in new) / max(1, samples):.0f} (prompt, once per episode) out {sum(r['n_out'] for r in new)}")
    return records


def describe(records: list[dict], by=("variant",)) -> str:
    s = summarize(records)
    lines = [
        f"n={s['n']} repair {s['repair_rate']:.3f} [{s['repair_lo']:.3f}, {s['repair_hi']:.3f}]  clean_ok {s['clean_ok']:.2f}  "
        f"valid {s['valid']:.2f}  noop_on_faulty {s['noop_faulty']:.2f}  reward {s['mean_reward']:.3f}  "
        f"mean out tokens {sum(r['n_out'] for r in records) / max(1, len(records)):.0f}"
    ]
    for key in by:
        for k, v in breakdown(records, key).items():
            lines.append(f"   {key}={k:18s} n={v['n']:4d} repair {v['repair_rate']:.2f} clean_ok {v['clean_ok']:.2f} valid {v['valid']:.2f} noop_f {v['noop_faulty']:.2f}")
    return "\n".join(lines)


async def _main():
    ap = argparse.ArgumentParser(description="Evaluate an LLM policy on gym episodes via Tinker")
    ap.add_argument("--split", help="frozen split file (data/*.jsonl)")
    ap.add_argument("--size", type=int, default=0, help="generate fresh episodes at this size when --split is not given")
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--model", default=INKLING)
    ap.add_argument("--model-path", default=None, help="tinker:// checkpoint; omit for the untrained base model")
    ap.add_argument("--effort", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--by", default="variant,fault,family")
    ap.add_argument("--direct", action="store_true", help="no-reasoning regime: prompt asks for an immediate answer")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--task", default="repair", choices=["repair", "decompile", "improve", "improve_loop"])
    a = ap.parse_args()
    if a.split:
        _, episodes = load_split(a.split)
        episodes = episodes[: a.n] if a.n else episodes
    else:
        cfg = GymConfig(size=a.size)
        episodes = [make_episode(a.seed0 + i, cfg) for i in range(a.n)]
    if a.task == "decompile":
        from .decompile import wrap

        episodes = [wrap(e) for e in episodes]
    if a.task in ("improve", "improve_loop"):
        from .imperfect import wrap as wrap_imperfect

        episodes = [wrap_imperfect(e, loop=a.task == "improve_loop") for e in episodes]
    policy = await LLMPolicy(tinker.ServiceClient(), a.model, a.model_path, a.effort, a.max_tokens, a.temperature).start()
    recs = await evaluate(policy, episodes, a.out, a.samples, a.concurrency, a.tag or os.path.basename(a.out), direct=a.direct)
    print(describe(recs, by=tuple(k for k in a.by.split(",") if k)))


if __name__ == "__main__":
    asyncio.run(_main())
