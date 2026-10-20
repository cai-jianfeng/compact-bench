#!/usr/bin/env python3
"""One-command entry point. Inputs are downloaded from HF as needed."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def load_env(path):
    """Load simple KEY=VALUE entries without executing shell expressions."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        name, separator, value = line.partition("=")
        if not separator or not name.strip().replace("_", "").isalnum():
            raise ValueError(f"Invalid environment entry in {path}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def launch(module, *args, round_number=None):
    env = dict(os.environ, PYTHONUTF8="1")
    env.pop("CB_S_ROUND", None)
    if round_number and round_number > 1:
        env["CB_S_ROUND"] = str(round_number)
    subprocess.run([sys.executable, "-m", module, *map(str, args)],
                   cwd=ROOT, env=env, check=True)


def task_track(args, C, ensure_data, out):
    import e1_runner as R
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_e1 import aggregate, write_report
    with open(ensure_data("e1/instances.json"), encoding="utf-8") as source:
        instances = json.load(source)
    if args.instance:
        wanted = set(args.instance)
        instances = [item for item in instances if item["instance_id"] in wanted]
        unknown = wanted - {item["instance_id"] for item in instances}
        if unknown:
            raise ValueError(f"Unknown instance IDs: {', '.join(sorted(unknown))}")
    if args.limit:
        instances = instances[:args.limit]
    C.RESULTS_ROOT = str(out)
    os.environ["CB_RESULTS_ROOT"] = str(out)
    failures = []
    for inst in instances:
        iid = inst["instance_id"]
        baseline_path = out / f"{iid}__baseline.json"
        try:
            if baseline_path.is_file():
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            else:
                print(f"[task] PRE {iid}", flush=True)
                root, venv = R.setup_instance(inst)
                f2p = R.resolve_f2p(inst, root)
                R.force_state(venv, root)
                base_pass, _ = R.run_tests(root, venv, f2p)
                if base_pass:
                    raise RuntimeError("Base repository unexpectedly passes FAIL_TO_PASS")
                H, _, directory, snapshot = R.phase_a_state(inst, root, venv)
                R.force_state(venv, directory)
                messages = list(H) + [{"role": "user", "content": "继续完成任务：确保修复生效并通过相关测试。"}]
                logs = []
                _, tools = R.episode(messages, directory, 8, logs, venv=venv)
                R.force_state(venv, directory)
                passed, error = R.run_tests(directory, venv, f2p)
                baseline = {"instance_id": iid, "pre_pass": passed, "f2p": f2p,
                            "tools": tools, "test_err": error, "logs_tail": logs[-3:]}
                C.save_json(str(baseline_path), baseline)
            for model in args.model:
                stem = f"{iid}__xb-{C.file_id(model)}.{args.prompt}"
                first = out / f"{stem}.json"
                if not first.is_file():
                    launch("harness.e1_xb", iid, model, args.prompt)
                result = json.loads(first.read_text(encoding="utf-8"))
                if args.rounds == 3 and result["post_pass"] != baseline["pre_pass"]:
                    for number in (2, 3):
                        if not (out / f"{stem}.r{number}.json").is_file():
                            launch("harness.e1_xb", iid, model, args.prompt, round_number=number)
        except Exception as exc:
            failures.append({"instance_id": iid, "error": str(exc)})
            print(f"[task] {iid}: {exc}", file=sys.stderr, flush=True)
    C.save_json(str(out / "errors.json"), failures)
    report = aggregate(out, args.prompt, [item["instance_id"] for item in instances])
    report["requested_instances"] = len(instances)
    report["errors"] = failures
    write_report(str(out), report)
    return not failures


def main(argv=None):
    load_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Compact-Bench; data downloads automatically from Hugging Face.")
    parser.add_argument("--model", nargs="+", default=None, help="summarizer model ID(s)")
    parser.add_argument("--track", choices=("all", "task", "retention", "production"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="limit task instances and retention sessions (0=all)")
    parser.add_argument("--instance", action="append", help="task instance ID; repeat to select multiple")
    parser.add_argument("--prompt", choices=("claude", "codex", "opencode", "gemini", "kimi"), default="claude")
    parser.add_argument("--rounds", type=int, choices=(1, 3), default=3, help="task verification rounds")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--dataset", default=os.environ.get("CB_HF_DATASET", "caijanfeng/compact-bench"))
    parser.add_argument("--revision", default=os.environ.get("CB_DATA_REVISION", "main"))
    parser.add_argument("--baseline-only", action="store_true", help="retention mechanical reference; no model calls")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.workers < 1:
        parser.error("--limit must be nonnegative and --workers must be positive")
    if args.baseline_only and args.track != "retention":
        parser.error("--baseline-only requires --track retention")
    if args.track != "production" and not args.model and not args.baseline_only:
        parser.error("--model is required for task and retention evaluation")
    args.model = args.model or ["anchor_index"]
    if args.track in ("all", "task"):
        if not sys.platform.startswith("linux"):
            parser.error("Task Preservation needs Linux. Use Docker Compose or WSL; retention/production also run natively.")
        for executable in ("patch", "git"):
            if not shutil.which(executable):
                parser.error(f"Task Preservation needs {executable}; use the supplied Docker image.")
    os.environ["CB_HF_DATASET"] = args.dataset
    os.environ["CB_DATA_REVISION"] = args.revision
    sys.path.insert(0, str(ROOT / "harness"))
    C = importlib.import_module("cb_common")
    ensure_data = importlib.import_module("cb_data").ensure_data
    if args.track != "production" and not args.baseline_only:
        for model in args.model:
            C.key_for_model(model)
        if args.track in ("all", "task"):
            C.key_for_model(C.AGENT_MODEL)
    # The hash prevents incompatible models/settings from reusing old results.
    config = {"models": args.model, "dataset": args.dataset, "revision": args.revision,
              "agent": C.AGENT_MODEL, "prompt": args.prompt,
              "summarizer_api": os.environ.get("CB_SUMMARIZER_API", "auto"),
              "summary_max_tokens": C.SUMMARIZER_MAX_TOKENS,
              "baseline_only": args.baseline_only,
              "data_source": os.environ.get("CB_DATA_MIRROR") or os.environ.get("CB_DATA_BASE"),
              "endpoints": [C.MESSAGES_URL, C.CHAT_URL, C.RESPONSES_URL],
              "implementation": hashlib.sha256(b"".join(
                  path.read_bytes() for path in sorted((ROOT / "harness").glob("*.py")))).hexdigest()}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() / run_id
    output.mkdir(parents=True, exist_ok=True)
    C.save_json(str(output / "run.json"), config)
    os.environ["CB_WORK_ROOT"] = str(ROOT / "work" / run_id)
    os.environ["CB_VENV_ROOT"] = str(ROOT / "work" / run_id / "venvs")
    C.WORK_ROOT = os.environ["CB_WORK_ROOT"]
    C.VENV_ROOT = os.environ["CB_VENV_ROOT"]
    ok = True
    if args.track in ("all", "production"):
        launch("harness.e3_score", "--out", output / "production")
    if args.track in ("all", "retention"):
        for model in args.model:
            options = ["--model", model, "--n", args.limit, "--workers", args.workers,
                       "--out", output / "retention"]
            if args.baseline_only:
                options.append("--baseline-only")
            launch("harness.e2_score", *options)
            report_path = output / "retention" / f"e2_report_{C.file_id(model)}.json"
            ok = not json.loads(report_path.read_text(encoding="utf-8"))["errors"] and ok
    if args.track in ("all", "task"):
        ok = task_track(args, C, ensure_data, output / "task") and ok
    print(f"\nResults: {output}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Compact-Bench: {exc}", file=sys.stderr)
        sys.exit(1)
