#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 anchor-fidelity track: compression prompts x one summarizer model plus
the mechanical anchor_index reference.

For every session, the model summarizes the serialized history once per
production prompt; recall is the fraction of the session's hindsight-verified
atoms that appear verbatim in the summary. anchor_index is the regex-only
mechanical upper bound (no LLM).

Usage:
    CB_API_KEY=... python3 -m harness.e2_score --model <model> [--n N] [--out results/e2]
Output:
    <out>/summaries_<model>.jsonl  (one record per session x prompt)
    <out>/e2_report_<model>.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_common as C
from cb_data import ensure_data

ORDER = ["claude", "codex", "opencode", "gemini", "kimi", "anchor_index"]


def compress_one(args):
    sid, client, tr, atoms, variant, model = args
    try:
        S = C.summarize(model, tr + "\n\n" + C.PROMPTS[variant], C.key_for_model(model))
    except Exception as e:
        return {"sample_id": sid, "client": client, "variant": variant, "error": str(e)[:150]}
    hits = sum(1 for a in atoms if a["value"] in S)
    return {"sample_id": sid, "client": client, "variant": variant,
            "S": S, "S_len": len(S), "recall": hits / len(atoms) if atoms else None,
            "n_atoms": len(atoms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="summarizer model id")
    ap.add_argument("--n", type=int, default=0, help="limit to first N sessions")
    ap.add_argument("--out", default=os.path.join(C.RESULTS_ROOT, "e2"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--baseline-only", action="store_true", help="run the mechanical reference without a model API")
    args = ap.parse_args()
    model = args.model

    samples = [json.loads(l) for l in open(ensure_data("e2/sessions.jsonl"), encoding="utf-8") if l.strip()]
    if args.n:
        samples = samples[: args.n]
    jobs = []
    for s in samples:
        tr = C.transcript(s["H"])
        for variant in ([] if args.baseline_only else C.PROMPTS):
            jobs.append((s["sample_id"], s["client"], tr, s["atoms"], variant, model))
    print(f"samples={len(samples)}, compress jobs={len(jobs)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model_file = C.file_id(model)
    out_path = os.path.join(args.out, f"summaries_{model_file}.jsonl")
    # resume: skip (sample, variant) pairs already present
    done_keys, results = set(), []
    if os.path.exists(out_path):
        selected = {sample["sample_id"] for sample in samples}
        for l in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # Recover a partial last record after interruption.
            if r["sample_id"] not in selected:
                continue
            if r.get("recall") is not None or "error" not in r:
                key = (r["sample_id"], r["variant"])
                if key not in done_keys:
                    done_keys.add(key)
                    results.append(r)
    jobs = [j for j in jobs if (j[0], j[4]) not in done_keys]
    print(f"resume: skip {len(done_keys)} done, {len(jobs)} to go", flush=True)

    # A newline separates any interrupted final record from new records.
    with open(out_path, "a", encoding="utf-8") as fout:
        if os.path.getsize(out_path):
            fout.write("\n")
        def emit(r):
            results.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()

        with ThreadPoolExecutor(args.workers) as ex:
            for r in ex.map(compress_one, jobs):
                emit(r)
        # mechanical anchor reference
        for s in samples:
            if (s["sample_id"], "anchor_index") in done_keys:
                continue
            S = C.anchor_baseline(s["H"])
            hits = sum(1 for a in s["atoms"] if a["value"] in S)
            emit({"sample_id": s["sample_id"], "client": s["client"], "variant": "anchor_index",
                  "S": S, "S_len": len(S), "recall": hits / s["n_atoms"], "n_atoms": s["n_atoms"]})

    ok = [r for r in results if r.get("recall") is not None]
    byv = defaultdict(list)
    for r in ok:
        byv[r["variant"]].append(r)
    md = f"# Recurring-Detail Retention ({model})\n\n"
    md += f"sessions: {len(samples)}\n\n"
    md += "| variant | mean recall | n | mean summary chars |\n|---|---:|---:|---:|\n"
    rows = []
    for v in ORDER:
        g = byv.get(v, [])
        if g:
            rows.append((v, sum(r["recall"] for r in g) / len(g), len(g),
                         sum(r["S_len"] for r in g) / len(g)))
    for v, m, n_, sl in sorted(rows, key=lambda x: -x[1]):
        md += f"| {v} | {m:.1%} | {n_} | {sl:.0f} |\n"
    md += "\n## per client\n\n| client | variant | mean recall | n |\n|---|---|---:|---:|\n"
    for c in sorted({r["client"] for r in ok}):
        for v in ORDER:
            g = [r for r in byv.get(v, []) if r["client"] == c]
            if g:
                md += f"| {c} | {v} | {sum(r['recall'] for r in g) / len(g):.1%} | {len(g)} |\n"
    md += "\nNote: anchor_index is regex-only extraction (no LLM), an approximate mechanical upper bound.\n"
    rep = os.path.join(args.out, f"e2_report_{model_file}.md")
    open(rep, "w", encoding="utf-8").write(md)
    C.save_json(os.path.join(args.out, f"e2_report_{model_file}.json"),
                {"model": model, "sessions": len(samples),
                 "errors": sum("error" in r for r in results),
                 "variants": {v: {"recall": m, "n": n, "mean_summary_chars": sl}
                              for v, m, n, sl in rows}})
    print(md, flush=True)
    print(f"-> {out_path}\n-> {rep}", flush=True)


if __name__ == "__main__":
    main()
