"""Join the per-GPU parts of one run into the single ``<ds>.json`` the tables read.

``run_metacog_grid.sh`` splits a dataset's categories across GPUs, one ``part<i>.json`` each.
Categories are fitted and scored independently, so the union of the parts is what one serial run
would have written. Parts whose configs differ are refused, since a table must not mix protocols.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def comparable(config: dict) -> dict:
    """The config without ``device``, which legitimately differs between workers."""
    return {k: v for k, v in config.items() if k != "device"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged: dict | None = None
    categories: dict = {}
    for path in args.parts:
        # A worker that never wrote is skipped, so the merge keeps whatever did finish.
        if not os.path.exists(path):
            print(f"  note: {path} does not exist, skipping")
            continue
        with open(path) as handle:
            part = json.load(handle)
        if merged is None:
            merged = part
        elif comparable(part["config"]) != comparable(merged["config"]):
            differing = {k for k in set(part["config"]) | set(merged["config"])
                         if part["config"].get(k) != merged["config"].get(k)} - {"device"}
            print(f"REFUSING to merge: configs differ on {sorted(differing)}", file=sys.stderr)
            return 1
        for name, result in part["categories"].items():
            if "arms" in result and name not in categories:
                categories[name] = result

    assert merged is not None, "none of the parts exist"
    merged["categories"] = dict(sorted(categories.items()))
    with open(args.output, "w") as handle:
        json.dump(merged, handle, indent=2)
    print(f"merged {len(categories)} categories -> {args.output}: {sorted(categories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
