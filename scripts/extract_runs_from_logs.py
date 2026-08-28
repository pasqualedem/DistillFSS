#!/usr/bin/env python
"""Extract finished-run records from out/<grid>/run_*.{yaml,log} into a CSV.

Emits the same flattened schema as the W&B download (results/runs.csv), so the
rows can be concatenated straight into that cache and consumed by build_long().
Pure stdlib + yaml on purpose: this has to run on the ReCaS frontend, where
importing torch gets the process SIGKILLed.

Usage: extract_runs_from_logs.py [--glob 'out/*Transfer_*'] [--out recas_runs.csv]
"""
import argparse
import csv
import glob
import os
import re

import yaml

# "INFO [...] [Refine] Test - MulticlassJaccardIndex_fg: 0.479..."  (Training - ... must NOT match)
FG_RE = re.compile(r"Test - MulticlassJaccardIndex_fg:\s*([0-9.eE+-]+)")

COLUMNS = [
    "id", "name", "state", "_timestamp",
    "model.name", "model.backbone",
    "model.params.teacher.name", "model.params.teacher.backbone",
    "refinement.lr", "refinement.hot_parameters", "refinement.max_iterations",
    "/MulticlassJaccardIndex_fg",
]


def flatten(d, pk="", sep="."):
    out = {}
    for k, v in (d or {}).items():
        nk = f"{pk}{sep}{k}" if pk else k
        if isinstance(v, dict):
            out.update(flatten(v, nk, sep))
        else:
            out[nk] = v
    return out


def parse_run(yaml_path):
    log_path = yaml_path[: -len(".yaml")] + ".log"
    if not os.path.exists(log_path):
        return None
    with open(log_path, errors="replace") as f:
        text = f.read()
    hits = FG_RE.findall(text)
    if not hits:
        return None                       # never reached test -> not a usable run
    fg = float(hits[-1])                  # last = final test evaluation

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", {}) or {}
    ref = cfg.get("refinement", {}) or {}
    rec = {
        "id": yaml_path,                  # stable synthetic id
        "name": os.path.basename(os.path.dirname(yaml_path)) + "/" +
                os.path.basename(yaml_path)[: -len(".yaml")],
        "state": "finished",
        "_timestamp": os.path.getmtime(log_path),
        "model.name": model.get("name"),
        "model.backbone": model.get("backbone"),
        "model.params.teacher.name": (model.get("params") or {}).get("teacher", {}).get("name")
            if isinstance((model.get("params") or {}).get("teacher"), dict) else None,
        "model.params.teacher.backbone": (model.get("params") or {}).get("teacher", {}).get("backbone")
            if isinstance((model.get("params") or {}).get("teacher"), dict) else None,
        "refinement.lr": ref.get("lr"),
        "refinement.hot_parameters": ref.get("hot_parameters"),
        "refinement.max_iterations": ref.get("max_iterations"),
        "/MulticlassJaccardIndex_fg": fg,
    }

    # dataset.datasets.test_<ds>.{prompt_images,seed,fold}
    for k, v in flatten(cfg.get("dataset", {}), "dataset").items():
        if ".test_" in k:
            rec[k] = v

    # Full model.*/refinement.* config. The aggregation groups runs by their whole
    # hyper-parameter fingerprint before averaging seeds; if these runs carry only the
    # 7 fields above while the W&B-side runs carry ~51, no log-recovered run can ever
    # match a W&B run, and seed replicates of the same experiment never pool.
    for section in ("model", "refinement"):
        for k, v in flatten(cfg.get(section, {}), section).items():
            rec.setdefault(k, v)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/*")
    ap.add_argument("--out", default="extracted_runs.csv")
    args = ap.parse_args()

    recs, skipped = [], 0
    for d in sorted(glob.glob(args.glob)):
        if not os.path.isdir(d):
            continue
        for y in sorted(glob.glob(os.path.join(d, "run_*.yaml"))):
            try:
                r = parse_run(y)
            except Exception as e:
                print(f"  !! {y}: {e!r}")
                r = None
            if r:
                recs.append(r)
            else:
                skipped += 1

    cols = list(COLUMNS)
    for r in recs:
        for k in r:
            if k not in cols:
                cols.append(k)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)
    print(f"extracted {len(recs)} runs with a test metric ({skipped} run files without one) -> {args.out}")


if __name__ == "__main__":
    main()
