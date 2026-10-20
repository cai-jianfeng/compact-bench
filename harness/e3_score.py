#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 drift track scorer (deterministic; no model calls).

Reads the frozen multi-compaction chains in tasks/e3/chains.jsonl. For every
boundary k of every chain:
  R_k    fraction of the atoms used in the following real requests (at most 5)
         that appear verbatim in the production summary S_k
  triplet 1 if a file already written before the boundary (from the preceding
         request's tool output) reappears in a pending/todo context of S_k

Usage:
    python3 -m harness.e3_score [--chains tasks/e3/chains.jsonl] [--out results/e3]
Output:
    <out>/e3_results.jsonl  (per-chain per-k measurements)
    <out>/e3_report.md
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cb_data import ensure_data

PENDING_RE = re.compile(r"(待办|待完成|还需要|尚未|未完|pending|to[- ]?do|next steps?|remaining)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=None)
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("CB_RESULTS_ROOT",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")),
        "e3"))
    args = ap.parse_args()
    chains_path = args.chains or ensure_data("e3/chains.jsonl")

    chains = [json.loads(l) for l in open(chains_path, encoding="utf-8") if l.strip()]
    results = []
    for ch in chains:
        per_k = []
        for pk in ch["per_k"]:
            S = pk["S_k"]
            atomsF = pk["atomsF"]
            nF = sum(len(v) for v in atomsF.values())
            inS = sum(1 for vs in atomsF.values() for v in vs if v in S)
            triplet = 0
            for f in pk.get("done_files", []):
                base = f.split("/")[-1]
                hit = False
                for m in PENDING_RE.finditer(S):
                    ctx = S[max(0, m.start() - 150): m.end() + 150]
                    if base in ctx:
                        hit = True
                        break
                if hit:
                    triplet += 1
            per_k.append({"k": pk["k"], "marker": pk["marker"], "form": pk["form"],
                          "n_atoms_F": nF, "R_k": inS / nF if nF else None,
                          "triplet_errors": triplet, "S_len": len(S)})
        if per_k:
            r1 = next((p["R_k"] for p in per_k if p["R_k"] is not None), None)
            last_r = per_k[-1]["R_k"]
            results.append({"sess": ch["sess"], "client": ch["client"], "n_bounds": ch["n_bounds"],
                            "per_k": per_k,
                            "ratio_last_first": (last_r / r1) if (r1 and last_r is not None) else None})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e3_results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_k, trip_k = defaultdict(list), defaultdict(list)
    for r in results:
        for p in r["per_k"]:
            if p["R_k"] is not None:
                by_k[p["k"]].append(p["R_k"])
            trip_k[p["k"]].append(p["triplet_errors"] > 0)
    rows = []
    for k in sorted(by_k):
        v = by_k[k]
        rows.append(f"| {k} | {sum(v) / len(v):.1%} | {len(v)} | {sum(trip_k[k]) / len(trip_k[k]):.1%} |")
    ratios = [r["ratio_last_first"] for r in results if r["ratio_last_first"] is not None]
    ratio_mean = sum(ratios) / len(ratios) if ratios else None
    md = f"""# Production-Chain Retention (n={len(results)})

| compaction index k | mean R_k | n | temporal-disorder rate |
|---|---:|---:|---:|
{chr(10).join(rows)}

- last/first retention ratio mean = {f"{ratio_mean:.2f}" if ratio_mean is not None else "N/A"} (n={len(ratios)})
- R_k = regex recall, in the production summary S_k, of atoms used in the at most 5 requests after boundary k; temporal disorder = a file already created/modified before the boundary appears in a pending context of S_k.
"""
    rep = os.path.join(args.out, "e3_report.md")
    open(rep, "w", encoding="utf-8").write(md)
    print(md, flush=True)
    print(f"-> {rep}", flush=True)


if __name__ == "__main__":
    main()
