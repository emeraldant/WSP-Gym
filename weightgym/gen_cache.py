"""Pre-generate episodes of a split into a JSONL cache (SGD-trained networks take ~0.2 s each).

    python -m weightgym.gen_cache train_trained 0 6000 cache/train_trained.jsonl
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

from .splits import episode_for


def _one(args):
    split, i = args
    return i, json.dumps(episode_for(split, i).to_json())


def main() -> None:
    split, start, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with Pool(3) as pool, open(out, "a") as f:
        for i, line in pool.imap(_one, [(split, i) for i in range(start, start + n)], chunksize=16):
            f.write(json.dumps({"index": i}) + "\t" + line + "\n")


def load_cache(path: str) -> dict:
    from .env import Episode

    out = {}
    with open(path) as f:
        for row in f:
            head, body = row.split("\t", 1)
            out[json.loads(head)["index"]] = Episode.from_json(json.loads(body))
    return out


if __name__ == "__main__":
    main()
