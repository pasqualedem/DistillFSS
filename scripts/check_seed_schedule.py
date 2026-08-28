#!/usr/bin/env python
"""Make the seed schedule of every canonical config reproduce our experiment exactly.

Seeds are the default now, so there is no separate seedvar tree: every config under
parameters/{baselines,distill,refine}/<model>/<Dataset>.yaml sweeps 5/3/2/2 seeds across
the four shot tiers -- the deterministic support set (seed: null) plus the reruns
0-3 / 0-1 / 0 / 0. Twelve runs per model x dataset cell.

The edit is textual, so hand-written recipes keep their comments and formatting: each
`seed: [...]` list in the seed schedule gains a leading null if it lacks one. Every
touched file is then expanded through the real create_experiment and checked to yield
exactly 5/3/2/2 for a single dataset.

    python scripts/check_seed_schedule.py
"""
import glob
import os
import re
import sys
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from distillfss.utils.grid import create_experiment  # noqa: E402

FAMILIES = ["baselines", "distill", "refine"]
WANT = [5, 3, 2, 2]
SEED_RE = re.compile(r"(seed:\s*\[)([^\]]*)(\])")


def add_null(text):
    """Prepend null to every rerun-seed list that does not already carry it."""
    def sub(m):
        items = [x.strip() for x in m.group(2).split(",") if x.strip()]
        if items and items[0] != "null":
            items = ["null"] + items
        return m.group(1) + ", ".join(items) + m.group(3)
    return SEED_RE.sub(sub, text)


def schedule(path):
    """{shots: n_seeds} and the set of dataset keys this config expands to."""
    doc = yaml.safe_load(open(path))
    runs = create_experiment({k: v for k, v in doc.items() if k != "grid"})
    per = defaultdict(set)
    for r in runs:
        k = list(r["dataset"]["datasets"])[0]
        per[k, r["dataset"]["datasets"][k]["prompt_images"]].add(
            r["dataset"]["datasets"][k]["seed"])
    return ({s: len(v) for (_, s), v in per.items()}, {k for k, _ in per}, len(runs))


def main():
    ok, fail, skipped = [], [], []
    for fam in FAMILIES:
        for path in sorted(glob.glob(os.path.join(ROOT, "parameters", fam, "*", "*.yaml"))):
            text = open(path).read()
            new = add_null(text)
            if new != text:
                open(path, "w").write(new)

            try:
                per_shot, dsk, n = schedule(path)
            except Exception as e:
                fail.append((path, f"expand failed: {e}"))
                continue
            counts = [per_shot[s] for s in sorted(per_shot)]
            if not counts:
                skipped.append(path)          # no seed sweep in this file
            elif len(dsk) != 1 or counts != WANT[:len(counts)]:
                fail.append((path, f"{per_shot} datasets={dsk} runs={n}"))
            else:
                ok.append((path, n))

    print("=" * 80)
    print(f"seed-aware configs at 5/3/2/2 : {len(ok)}")
    print(f"no seed sweep (left alone)    : {len(skipped)}")
    runs = sorted({n for _, n in ok})
    print(f"runs per config               : {runs}")
    if fail:
        print(f"\nFAILURES ({len(fail)}):")
        for p, why in fail:
            print(f"  {os.path.relpath(p, ROOT)}: {why}")


if __name__ == "__main__":
    main()
