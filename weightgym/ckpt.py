"""Checkpoint housekeeping on Tinker.

An Inkling 2 Small LoRA training state is ~50 GB and its sampler weights ~17 GB, so a run that
saves often fills the account's free checkpoint storage within a few saves. Training code calls
``delete_paths`` to drop superseded checkpoints; this module only ever deletes paths it is given.

    python -m weightgym.ckpt list
    python -m weightgym.ckpt delete tinker://... tinker://...
"""

from __future__ import annotations

import asyncio
import sys

import tinker


async def list_checkpoints() -> list:
    rc = tinker.ServiceClient().create_rest_client()
    return list((await rc.list_user_checkpoints_async(limit=500)).checkpoints)


async def delete_paths(paths: list[str], rest_client=None) -> list[str]:
    """Delete the given tinker:// checkpoint paths. Returns the ones actually deleted."""
    rc = rest_client or tinker.ServiceClient().create_rest_client()
    done = []
    for p in paths:
        try:
            await rc.delete_checkpoint_from_tinker_path_async(p)
            done.append(p)
        except Exception as e:  # noqa: BLE001 - already gone or not ours: report, keep going
            print(f"[ckpt] could not delete {p}: {e!r}"[:300])
    return done


async def _main(argv: list[str]) -> None:
    if argv[:1] == ["delete"]:
        print("deleted:", await delete_paths(argv[1:]))
    total = 0
    for c in await list_checkpoints():
        size = getattr(c, "size_bytes", 0) or 0
        total += size
        print(f"{c.tinker_path}  {getattr(c, 'checkpoint_type', '')}  {size / 1e9:.1f} GB  expires {getattr(c, 'expires_at', None)}")
    print(f"total {total / 1e9:.1f} GB")


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
