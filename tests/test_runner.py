import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "scripts"))
import cb_common as C
import e1_runner as R
from score_e1 import aggregate


@contextlib.contextmanager
def fixture_server(files, response=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            body = files.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", request))
            body = json.dumps(response or {"content": [{"type": "text", "text": ""}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_base = Path(tempfile.gettempdir()).resolve()
        self.temp = Path(tempfile.mkdtemp(prefix="compact-bench-test-", dir=self.temp_base)).resolve()
        self.assertEqual(self.temp.parent, self.temp_base)

    def tearDown(self):
        self.assertEqual(self.temp.resolve().parent, self.temp_base)
        shutil.rmtree(self.temp)

    def cli(self, base, *arguments):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("CB_") and k not in ("HF_TOKEN",)}
        env.update(CB_DATA_BASE=base, CB_DATA_CACHE=str(self.temp / "cache"),
                   CB_API_KEY="synthetic-test-value", CB_MESSAGES_URL=base + "/messages",
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run.py"),
                                 "--output", str(self.temp / "results"), *arguments],
                                cwd=self.temp, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_production_first_download_and_cached_rerun(self):
        chain = {"sess": "toy", "client": "toy", "n_bounds": 2,
                 "per_k": [{"k": k, "marker": "toy", "form": "embed",
                            "S_k": "src/main.py", "atomsF": {"path": ["src/main.py"]},
                            "done_files": []} for k in (1, 2)]}
        with fixture_server({"/e3/chains.jsonl": (json.dumps(chain) + "\n").encode()}) as (base, requests):
            first = self.cli(base, "--track", "production")
            second = self.cli(base, "--track", "production")
            self.assertEqual(sum(method == "GET" for method, _ in requests), 1)
            self.assertIn("100.0%", first.stdout)
            self.assertNotIn("[data]", first.stdout)
            self.assertNotIn("[data]", second.stderr)

    def test_retention_api_empty_summary_and_resume_subset(self):
        samples = [{"sample_id": str(i), "client": "toy",
                    "H": [{"role": "user", "content": "src/main.py"}],
                    "atoms": [{"kind": "path", "value": "src/main.py"}], "n_atoms": 1}
                   for i in range(2)]
        payload = "".join(json.dumps(s) + "\n" for s in samples).encode()
        with fixture_server({"/e2/sessions.jsonl": payload}) as (base, requests):
            self.cli(base, "--track", "retention", "--model", "provider/model", "--limit", "2")
            self.cli(base, "--track", "retention", "--model", "provider/model", "--limit", "1")
            self.assertEqual(sum(method == "POST" for method, _ in requests), 10)
            report = json.loads(next((self.temp / "results").glob("*/retention/e2_report_*.json")).read_text())
            self.assertEqual(report["sessions"], 1)
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["variants"]["claude"]["n"], 1)
            self.assertEqual(report["variants"]["claude"]["recall"], 0)
            self.assertEqual(report["variants"]["anchor_index"]["recall"], 1)

    def test_responses_json_protocol(self):
        response = {"status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "summary"}]}]}
        with fixture_server({}, response) as (base, requests):
            with patch.object(C, "RESPONSES_URL", base + "/responses"):
                self.assertEqual(C.call_responses("toy", "input", "unused", tries=1), "summary")
            self.assertIs(requests[0][1]["stream"], False)

    def test_incomplete_stream_is_not_accepted(self):
        raw = ('data: {"type":"response.output_text.delta","delta":"partial"}\n'
               'data: {"type":"response.incomplete","response":{"status":"incomplete"}}\n').encode()
        with patch.object(C.urllib.request, "urlopen", return_value=io.BytesIO(raw)), patch.object(C.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                C.call_responses("toy", "input", "unused", tries=1)

    def test_task_pending_rounds_and_prompt_isolation(self):
        iid = "toy"
        C.save_json(str(self.temp / f"{iid}__baseline.json"), {"instance_id": iid, "pre_pass": True})
        C.save_json(str(self.temp / f"{iid}__xb-test.claude.json"),
                    {"instance_id": iid, "variant": "claude", "post_pass": False})
        C.save_json(str(self.temp / f"{iid}__xb-other.codex.json"),
                    {"instance_id": iid, "variant": "codex", "post_pass": False})
        report = aggregate(self.temp)
        self.assertEqual(set(report["models"]), {"test"})
        self.assertIsNone(report["models"]["test"]["stable_rate"])
        for number, passed in ((2, False), (3, True)):
            C.save_json(str(self.temp / f"{iid}__xb-test.claude.r{number}.json"),
                        {"instance_id": iid, "variant": "claude", "post_pass": passed})
        self.assertEqual(aggregate(self.temp)["models"]["test"]["stable_rate"], 1)

    def test_read_file_rejects_sibling(self):
        repo = self.temp / "repo"
        sibling = self.temp / "repo-other"
        repo.mkdir()
        sibling.mkdir()
        (sibling / "marker").write_text("synthetic marker")
        self.assertEqual(R.read_file("../repo-other/marker", str(repo)), "blocked: path escape")

    def test_xb_loads_state_before_summary(self):
        import e1_xb as X
        manifest = self.temp / "instances.json"
        manifest.write_text(json.dumps([{"instance_id": "toy"}]))
        snapshot = self.temp / "snapshot"
        snapshot.mkdir()
        events = []

        def state(*args):
            events.append("state")
            return [{"role": "user", "content": "new"}], [], str(self.temp / "D"), str(snapshot)

        def compress(history, *args):
            events.append(history[0]["content"])
            return "summary"

        with patch.object(X, "WORK", str(self.temp)), \
             patch.object(C, "RESULTS_ROOT", str(self.temp)), \
             patch.object(X, "ensure_data", return_value=str(manifest)), \
             patch.object(R, "setup_instance", return_value=("root", "venv")), \
             patch.object(R, "resolve_f2p", return_value=["test"]), \
             patch.object(R, "phase_a_state", state), \
             patch.object(X, "compress_x", compress), \
             patch.object(R, "force_state"), \
             patch.object(R, "episode", return_value=([], [])), \
             patch.object(R, "run_tests", return_value=(True, "")), \
             patch.object(sys, "argv", ["runner", "toy", "model"]), \
             contextlib.redirect_stdout(io.StringIO()):
            X.main()
        self.assertEqual(events, ["state", "new"])

    def test_data_export_redacts_same_credential_in_summary_and_label(self):
        from prepare_data import redact_records, PROVIDER_KEY
        token = "sk-" + "0123456789abcdef" * 3
        records, counts = redact_records([
            {"S_k": "credential=" + token, "atomsF": {"kv": ["API_KEY=" + token]}}])
        row = records[0]
        self.assertNotIn(token, json.dumps(row))
        self.assertFalse(PROVIDER_KEY.search(json.dumps(row)))
        self.assertEqual(row["S_k"].split("=", 1)[1], row["atomsF"]["kv"][0].split("=", 1)[1])
        self.assertGreater(counts["credential"], 0)


if __name__ == "__main__":
    unittest.main()
