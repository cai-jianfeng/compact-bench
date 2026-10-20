#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 counterfactual-replay runner (v3 protocol).

For one instance: set up the repository at its base commit, obtain the frozen
phase-A state (trajectory H3 + working-directory snapshot), then run the POST
continuation from summary S and judge with the instance's FAIL_TO_PASS tests.

Frozen histories and snapshots are downloaded automatically as pairs. A missing
snapshot is an error unless CB_REGENERATE_STATE=1 explicitly enables fresh phase A.

Usage:
    python3 -m harness.e1_runner <instance_id> [variant=claude]
Output:
    <results_root>/<instance_id>__<variant>.v3.json
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_common as C
from cb_data import ensure_data

WORK = C.WORK_ROOT

TOOLS_SPEC = [
    {"name": "read_file", "description": "读取仓库内文件", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
    {"name": "run_command", "description": "在仓库根目录执行命令（pytest/grep/sed/cat/ls/python 等）", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]
ALLOWED = ("pytest", "python", "ls", "grep", "rg", "cat", "sed", "head", "tail", "find", "echo",
           "cd", "pwd", "wc", "diff", "patch", "git", "mkdir", "cp", "mv", "touch", "pip", "rm")

_SECRET_RE = re.compile(r"(LLM__|API_?KEY|ACCESS_?TOKEN|SECRET|TOKEN)", re.I)


def _scrubbed_env():
    """Execution environment for model-generated commands: strip every
    credential-looking variable; pytest/git/pip need none of them."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_RE.search(k)}
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST"] = "7.99.0"
    return env


def checked(command, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env or _scrubbed_env(),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Environment command failed: {command[0]}\n"
                           + (result.stdout + result.stderr)[-3000:])
    return result


def extract_archive(archive, destination):
    """Allow internal repository links while rejecting extraction outside it."""
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Task extraction requires a current Python security release (or the supplied Docker image)")
    base = Path(destination).resolve()
    def checked_member(member, path):
        target = (base / member.name).resolve()
        if Path(member.name).is_absolute() or (target != base and base not in target.parents):
            raise ValueError("Unsafe archive member")
        # Evaluate each member at extraction time so earlier links cannot turn
        # a later member into an escape. The standard data filter also checks
        # link destinations and rejects device nodes and other special files.
        return tarfile.data_filter(member, path)

    with tarfile.open(archive) as source:
        source.extractall(destination, filter=checked_member)


def run_cmd(cmd, cwd, timeout=120, venv=None):
    first = cmd.strip().split()[0] if cmd.strip() else ""
    if first not in ALLOWED:
        return f"blocked: command '{first}' not allowed"
    env = _scrubbed_env()
    if venv:
        # the agent shares the judge's venv: PATH points at venv/bin and
        # PYTHONPATH at the working tree, so flat-layout repos import the live
        # tree rather than a stale site-packages copy
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = os.path.abspath(cwd)
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = (r.stdout + r.stderr)[-3000:]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "blocked: timeout"


def read_file(path, cwd):
    base = Path(cwd).resolve()
    p = (base / path).resolve()
    if p != base and base not in p.parents:
        return "blocked: path escape"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:8000]
    except Exception as e:
        return f"error: {e}"


def _call_agent_tools(msgs):
    """Messages call with tool spec; returns the raw response dict."""
    import json as _json
    import urllib.request as _u
    body = {"model": C.AGENT_MODEL, "max_tokens": C.CONT_MAX_TOKENS,
            "messages": msgs, "tools": TOOLS_SPEC}
    if C.AGENT_MODEL.startswith(("kimi", "vendor/kimi", "internal/kimi")):
        body["thinking"] = {"type": "disabled"}
    req = _u.Request(C.MESSAGES_URL, data=_json.dumps(body).encode(),
                     headers={"x-api-key": C.key_for_model(C.AGENT_MODEL),
                              "anthropic-version": "2023-06-01",
                              "Content-Type": "application/json"})
    return C.post_with_retry(req)


def episode(msgs, cwd, max_steps, log, venv=None):
    """Real tool loop; returns (msgs, tool_log)."""
    tlog = []
    for step in range(max_steps):
        d = _call_agent_tools(msgs)
        content = d.get("content") or []
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        tus = [p for p in content if p.get("type") == "tool_use"]
        log.append({"step": step, "text": text[:500], "tools": [t.get("name") for t in tus]})
        if not tus:
            msgs.append({"role": "assistant", "content": content})
            break
        msgs.append({"role": "assistant", "content": content})
        results = []
        for tu in tus:
            name, inp = tu.get("name"), tu.get("input") or {}
            if name == "read_file":
                out = read_file(inp.get("file_path", ""), cwd)
                tlog.append(("read", inp.get("file_path", "")))
            else:
                out = run_cmd(inp.get("command", ""), cwd, venv=venv)
                tlog.append(("cmd", (inp.get("command") or "")[:200]))
            results.append({"type": "tool_result", "tool_use_id": tu.get("id"), "content": out})
        msgs.append({"role": "user", "content": results})
    return msgs, tlog


def setup_instance(inst):
    """Fetch + extract the source archive at base_commit, apply test_patch,
    build the per-instance venv (with per-repo era pins). Returns (root, venv)."""
    iid = inst["instance_id"]
    os.makedirs(os.path.join(WORK, "archives"), exist_ok=True)
    tgz = os.path.join(WORK, "archives", f"{iid.split('__')[1]}.tgz" if "__" in iid else f"{iid}.tgz")
    if not os.path.exists(tgz):
        url = f"https://codeload.github.com/{inst['repo']}/tar.gz/{inst['base_commit']}"
        print(f"[setup] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tgz)
    work = os.path.join(WORK, "work", iid)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    extract_archive(tgz, work)
    root = os.path.join(work, os.listdir(work)[0])
    patch_path = os.path.join(WORK, "work", f"{iid}.test.patch")
    with open(patch_path, "w", encoding="utf-8") as output:
        output.write(inst["test_patch"])
    checked(["patch", "-p1", "-i", os.path.abspath(patch_path)], cwd=root)

    venv = os.path.join(C.VENV_ROOT, iid)
    pybin = os.environ.get("CB_PYTHON", sys.executable)
    # per-repository era pins: xarray needs old numpy/pandas plus a preinstalled
    # setuptools_scm; pylint tests need pylib; sphinx 4.x needs sphinxcontrib-* <2
    REPO_EXTRA = {
        "pydata/xarray": ["setuptools_scm<8", "numpy<2", "pandas<2"],
        "pylint-dev/pylint": ["py"],
        "sphinx-doc/sphinx": ["docutils<0.18", "sphinxcontrib-applehelp==1.0.2", "sphinxcontrib-devhelp==1.0.2",
                          "sphinxcontrib-htmlhelp==2.0.0", "sphinxcontrib-qthelp==1.0.3", "sphinxcontrib-serializinghtml==1.1.5"],
    }
    extra = REPO_EXTRA.get(inst["repo"], [])

    def _build_venv():
        shutil.rmtree(venv, ignore_errors=True)
        checked([pybin, "-m", "venv", "--copies", venv])
        pip = os.path.join(venv, "bin", "pip")
        if extra:
            checked([pip, "install", "-q"] + extra)
        checked([pip, "install", "-q", root])
        checked([pip, "install", "-q", "pytest<8", "werkzeug<3", "urllib3<2"])
        if extra:
            # the repo's declared dependencies may upgrade the pins (sphinx 4.2
            # wants htmlhelp>=2); force them back after installation
            checked([pip, "install", "-q", "--no-deps", "--force-reinstall"] + extra)

    py = os.path.join(venv, "bin", "python")
    # a venv left half-built by a killed run would fail every test; verify
    # pytest imports, rebuild from scratch otherwise
    for _try in range(3):
        if _try > 0 or not os.path.exists(py):
            _build_venv()
        if subprocess.run([py, "-c", "import pytest"], capture_output=True).returncode == 0:
            return root, venv
    raise RuntimeError(f"venv broken for {iid}: pytest not importable after rebuilds")


def run_tests(root, venv, f2p):
    py = os.path.join(venv, "bin", "python")
    for t in f2p:
        r = subprocess.run([py, "-m", "pytest", "-o", "filterwarnings=ignore::DeprecationWarning",
                            "-x", "-q", t], cwd=root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300, env=_scrubbed_env())
        output = r.stdout + r.stderr
        if r.returncode not in (0, 1) or "ERROR collecting" in output or "ERROR at setup" in output:
            raise RuntimeError("Test environment failed:\n" + output[-1500:])
        if r.returncode != 0:
            return False, (r.stdout + r.stderr)[-500:]
    return True, ""


def force_state(venv, root):
    pip = os.path.join(venv, "bin", "pip")
    checked([pip, "install", "-q", "--force-reinstall", "--no-deps", root])


def resolve_f2p(inst, root0):
    """Bare test function names (sympy family) are resolved to node ids against
    the test files touched by test_patch (already applied in root0)."""
    f2p = json.loads(inst["FAIL_TO_PASS"]) if isinstance(inst["FAIL_TO_PASS"], str) else inst["FAIL_TO_PASS"]
    _tfiles = re.findall(r"diff --git a/(\S+\.py) ", inst["test_patch"])

    def _resolve(t):
        if t.endswith(".py") or "/" in t or "::" in t:
            return t
        for f in _tfiles:
            p = os.path.join(root0, f)
            if os.path.exists(p) and re.search(rf"def {re.escape(t)}\s*\(", open(p, errors="ignore").read()):
                return f"{f}::{t}"
        return f"{_tfiles[0]}::{t}" if _tfiles else t

    return [_resolve(t) for t in f2p]


def _load_H(iid):
    """Frozen phase-A trajectory: local regeneration cache first, then the
    dataset's frozen H3."""
    local = os.path.join(WORK, "H3", f"{iid}.H3.json")
    p = local if os.path.exists(local) else ensure_data(f"e1/H3/{iid}.H3.json")
    with open(p, encoding="utf-8") as source:
        cached = json.load(source)
    return cached["H"], cached["tlog_a"]


def phase_a_state(inst, root0, venv):
    """Return (H, tlog_a, D, snap): the frozen phase-A trajectory and working
    directory. Precedence: local snapshot -> dataset snapshot -> fresh phase A
    with CB_AGENT_MODEL (which then writes its own H3 cache and snapshot)."""
    iid = inst["instance_id"]
    D = os.path.join(WORK, "work", f"{iid}.D3")
    snap = os.path.join(WORK, "work", f"{iid}.D3snap")
    if os.path.exists(snap):
        H, tlog_a = _load_H(iid)
        shutil.rmtree(D, ignore_errors=True)
        shutil.copytree(snap, D, symlinks=False)
        return H, tlog_a, D, snap
    # dataset snapshot: per-instance tar.gz in the HF dataset repo
    try:
        stgz = ensure_data(f"e1/d3snap/{iid}.tar.gz")
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        extract_archive(stgz, os.path.dirname(snap))
        H, tlog_a = _load_H(iid)
        shutil.rmtree(D, ignore_errors=True)
        shutil.copytree(snap, D, symlinks=False)
        return H, tlog_a, D, snap
    except Exception as e:
        if os.environ.get("CB_REGENERATE_STATE") != "1":
            raise RuntimeError(f"Cannot load frozen state for {iid}: {e}") from e
        print(f"[phaseA] no dataset snapshot for {iid} ({type(e).__name__}); running phase A fresh", flush=True)
    # fresh phase A: copy the pristine repo into D, run the agent, snapshot
    shutil.rmtree(D, ignore_errors=True)
    shutil.copytree(root0, D, symlinks=False, ignore=shutil.ignore_patterns(".venv"))
    msgs = [{"role": "user", "content":
             f"你在仓库 {inst['repo']} 中修复以下问题。只能用 read_file/run_command 工具。"
             f"先定位相关代码再修改。问题：\n\n{inst['problem_statement']}"}]
    logs = []
    msgs, tlog_a = episode(msgs, D, 10, logs, venv=venv)
    H = msgs
    h_local = os.path.join(WORK, "H3", f"{iid}.H3.json")
    os.makedirs(os.path.dirname(h_local), exist_ok=True)
    C.save_json(h_local, {"H": H, "tlog_a": tlog_a})
    shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(D, snap, symlinks=False)
    return H, tlog_a, D, snap


def compress(H_msgs, variant="claude"):
    user = C.build_transcript(H_msgs) + "\n\n" + C.PROMPTS[variant]
    return C.summarize(C.AGENT_MODEL, user, C.key_for_model(C.AGENT_MODEL), max_tokens=6144)


def main():
    iid, variant = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "claude")
    picked = json.load(open(ensure_data("e1/instances.json"), encoding="utf-8"))
    inst = next(x for x in picked if x["instance_id"] == iid)
    root0, venv = setup_instance(inst)
    f2p = resolve_f2p(inst, root0)
    print(f"[setup] {iid} at {root0}", flush=True)
    ok, err = run_tests(root0, venv, f2p)
    print(f"[sanity] base FAIL_TO_PASS pass={ok} (expect False) {err[:120]}", flush=True)
    if ok:
        raise RuntimeError("Base repository unexpectedly passes FAIL_TO_PASS")

    # v3 protocol: phase A and both continuation conditions share one directory
    # D (production semantics: the repository path does not change across
    # compaction); the snapshot restores the phase-A end state between arms.
    H, tlog_a, D, snap = phase_a_state(inst, root0, venv)
    S = compress(H, variant)
    print(f"[phaseA] tools={len(tlog_a)} S_len={len(S)}", flush=True)

    results = {"instance_id": iid, "variant": variant, "proto": "v3", "f2p": f2p,
               "phaseA_tools": tlog_a, "base_pass": ok, "arms": {}}
    for arm in ("PRE", "POST"):
        if arm == "POST":
            # PRE modified D; restore the phase-A end state first
            shutil.rmtree(D, ignore_errors=True)
            shutil.copytree(snap, D, symlinks=False)
        if arm == "PRE":
            m = [dict(x) for x in H]
        else:
            m = [{"role": "user", "content": "The conversation so far has been compacted. What follows is the summary:\n\n" + S}]
        m.append({"role": "user", "content": "继续完成任务：确保修复生效并通过相关测试。"})
        arm_logs = []
        force_state(venv, D)
        m2, tlog = episode(m, D, 8, arm_logs, venv=venv)
        force_state(venv, D)
        passed, terr = run_tests(D, venv, f2p)
        # v1.5: file paths appearing in phase-A commands also count as "read"
        a_reads = {v for k, v in tlog_a if k == "read"}
        for k, v in tlog_a:
            if k == "cmd":
                for mm in re.finditer(r"([\w@~./-]+\.[a-zA-Z]{1,10})", v):
                    a_reads.add(mm.group(1).lstrip("/"))

        def _is_redo(path):
            p = (path or "").lstrip("/")
            return any(p.endswith(x) or x.endswith(p) or p == x for x in a_reads if x)

        redo = sum(1 for k, v in tlog if k == "read" and (_is_redo(v) or ("read", v) in set(tlog_a)))
        results["arms"][arm] = {"pass": passed, "tools": tlog, "redo_reads": redo,
                                "logs_tail": arm_logs[-3:], "test_err": terr[:300]}
        print(f"[{arm}] pass={passed} redo={redo}", flush=True)

    os.makedirs(C.RESULTS_ROOT, exist_ok=True)
    out = os.path.join(C.RESULTS_ROOT, f"{iid}__{variant}.v3.json")
    results["S"] = S
    C.save_json(out, results)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
