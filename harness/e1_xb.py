#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 cross-backbone cell runner: measure one summarizer model on one instance.

The summarizer under test produces summary S from the frozen phase-A
trajectory H3 (cached per instance and shared by all models); the POST
continuation then runs with the fixed agent CB_AGENT_MODEL from the restored
snapshot, and the instance's FAIL_TO_PASS tests judge the outcome.

Usage:
    python3 -m harness.e1_xb <instance_id> <summarizer_model> [variant=claude]
    CB_S_ROUND=2 python3 -m harness.e1_xb <instance_id> <model>   # K=3 re-seed
Output:
    <results_root>/<instance_id>__xb-<model>.<variant>[.r2|.r3].json
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_common as C
import e1_runner as R
from cb_data import ensure_data

WORK = C.WORK_ROOT


def compress_x(H_msgs, backbone, variant):
    user = C.build_transcript(H_msgs) + "\n\n" + C.PROMPTS[variant]
    return C.summarize(backbone, user, C.key_for_model(backbone))


def main():
    iid, backbone = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "claude"
    tag = f"xb-{C.file_id(backbone)}"
    with open(ensure_data("e1/instances.json"), encoding="utf-8") as source:
        picked = json.load(source)
    inst = next(x for x in picked if x["instance_id"] == iid)
    root0, venv = R.setup_instance(inst)
    f2p = R.resolve_f2p(inst, root0)

    # Obtain the paired history and repository state before generating S.
    H, _, D, snap = R.phase_a_state(inst, root0, venv)

    os.makedirs(os.path.join(WORK, "work"), exist_ok=True)
    s_cache = os.path.join(WORK, "work", f"{iid}.S.{tag}.{variant}{C.S_SFX}.txt")
    if os.path.exists(s_cache):
        with open(s_cache, encoding="utf-8") as source:
            S = source.read()
    else:
        S = compress_x(H, backbone, variant)
        with open(s_cache, "w", encoding="utf-8") as output:
            output.write(S)
    print(f"[xb] {iid} backbone={backbone} variant={variant} S_len={len(S)}", flush=True)

    # POST arm only: the PRE control is the shared first-seed baseline from the
    # main track; the snapshot restores the phase-A end state before continuing
    shutil.rmtree(D, ignore_errors=True)
    shutil.copytree(snap, D, symlinks=False)
    m = [{"role": "user", "content": "The conversation so far has been compacted. What follows is the summary:\n\n" + S}]
    m.append({"role": "user", "content": "继续完成任务：确保修复生效并通过相关测试。"})
    arm_logs = []
    R.force_state(venv, D)
    m2, tlog = R.episode(m, D, 8, arm_logs, venv=venv)
    R.force_state(venv, D)
    passed, terr = R.run_tests(D, venv, f2p)
    out = {"instance_id": iid, "backbone": backbone, "variant": variant, "proto": "v3-xb",
           "f2p": f2p, "post_pass": passed, "tools": tlog, "test_err": terr[:300],
           "logs_tail": arm_logs[-3:], "S": S, "round": int(C.S_ROUND or "1")}
    os.makedirs(C.RESULTS_ROOT, exist_ok=True)
    op = os.path.join(C.RESULTS_ROOT, f"{iid}__{tag}.{variant}{C.S_SFX}.json")
    C.save_json(op, out)
    print(f"[done] post_pass={passed} -> {op}", flush=True)


if __name__ == "__main__":
    main()
