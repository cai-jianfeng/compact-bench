#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate E1 results into the backbone damage table (r1, K=3-stable, Wilson CI).

Reads <results_root>:
  <iid>__<variant>.v3.json            main-track cell (both arms; PRE = shared control)
  <iid>__xb-<model>.<variant>.json    cross-backbone r1 cell (POST only)
  <iid>__xb-<model>.<variant>.r2/.r3  K=3 re-seed cells

Damage = PRE passes and POST fails. Stable damage = POST fails in at least two
of the three seeds. The two defective instances (pytest-10051, pytest-7236,
full-history continuations fail in three independent environments) are
excluded from the rates.

Usage:
    python3 scripts/score_e1.py [--results results] [--variant claude]
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "harness"))
from cb_data import ensure_data
import cb_common as C

EXCLUDE = {"pytest-dev__pytest-10051", "pytest-dev__pytest-7236"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate(results_dir, variant="claude", instance_ids=None):
    """Score existing results; pending verification never becomes zero damage."""
    records, baselines = {}, {}
    allowed = set(instance_ids) if instance_ids is not None else None
    for path in sorted(Path(results_dir).glob("*.json")):
        if "__xb-" not in path.name and not path.name.endswith(("__baseline.json", f"__{variant}.v3.json")):
            continue
        with path.open(encoding="utf-8") as source:
            row = json.load(source)
        iid = row.get("instance_id") or path.name.split("__xb-")[0].split(f"__{variant}")[0].split("__baseline")[0]
        if iid in EXCLUDE or (allowed is not None and iid not in allowed):
            continue
        if "__xb-" not in path.name:
            pre = row.get("pre_pass")
            if pre is None:
                pre = row.get("arms", {}).get("PRE", {}).get("pass")
            if pre is not None:
                # A dedicated baseline takes precedence over legacy main runs.
                if iid not in baselines or path.name.endswith("__baseline.json"):
                    baselines[iid] = pre
            continue
        suffix = path.name.split("__xb-", 1)[1]
        match = re.fullmatch(r"(.+)\." + re.escape(variant) + r"(?:\.r([23]))?\.json", suffix)
        if not match or row.get("variant", variant) != variant:
            continue
        model = row.get("backbone", match.group(1))
        round_number = int(row.get("round", match.group(2) or "1"))
        records.setdefault(model, {}).setdefault(iid, {})[round_number] = row.get("post_pass")
    output = {"variant": variant, "baseline_count": len(baselines), "models": {}}
    for model, cases in records.items():
        counts = dict(preserved=0, damaged=0, rescued=0, unsolved=0,
                      stable_damage=0, pending_verification=0, missing_first_round=0)
        denominator = sum(value is True for value in baselines.values())
        for iid, pre in baselines.items():
            rounds = cases.get(iid, {})
            post = rounds.get(1)
            if post is None:
                counts["missing_first_round"] += 1
                continue
            category = ("preserved" if post else "damaged") if pre else ("rescued" if post else "unsolved")
            counts[category] += 1
            if pre and post is False:
                if any(rounds.get(n) is None for n in (2, 3)):
                    counts["pending_verification"] += 1
                elif sum(rounds[n] is False for n in (1, 2, 3)) >= 2:
                    counts["stable_damage"] += 1
        complete = not counts["pending_verification"] and not counts["missing_first_round"]
        counts["pre_pass_n"] = denominator
        counts["stable_rate"] = counts["stable_damage"] / denominator if complete and denominator else None
        counts["stable_wilson_95"] = list(wilson(counts["stable_damage"], denominator)) if complete and denominator else None
        output["models"][model] = counts
    return output


def write_report(results_dir, report):
    lines = ["# Task Preservation", "",
             "| model | preserved | damaged | rescued | unsolved | stable damage | pending verification | missing r1 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for model, row in report["models"].items():
        stable = (f"{row['stable_damage']}/{row['pre_pass_n']}"
                  if row["stable_rate"] is not None else "pending")
        lines.append(f"| {model} | {row['preserved']} | {row['damaged']} | "
                     f"{row['rescued']} | {row['unsolved']} | {stable} | "
                     f"{row['pending_verification']} | {row['missing_first_round']} |")
    text = "\n".join(lines) + "\n"
    C.save_json(os.path.join(results_dir, "task_report.json"), report)
    with open(os.path.join(results_dir, "task_report.md"), "w", encoding="utf-8") as output:
        output.write(text)
    print(text)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("CB_RESULTS_ROOT", os.path.join(REPO, "results")))
    ap.add_argument("--variant", default="claude")
    ap.add_argument("--prepass", default=None,
                    help="optional fixed instance cohort; default uses saved PRE results")
    args = ap.parse_args()
    cells = None
    if args.prepass:
        with open(args.prepass, encoding="utf-8") as source:
            cells = [line.strip() for line in source if line.strip()]
    os.makedirs(args.results, exist_ok=True)
    write_report(args.results, aggregate(args.results, args.variant, cells))


if __name__ == "__main__":
    main()
