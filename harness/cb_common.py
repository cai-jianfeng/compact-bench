#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact-Bench harness shared core: configuration, LLM transport, the five
production compaction prompts, transcript construction and atom extraction.

Pure standard library. No credentials, endpoints, or run state are stored in
this file; everything is configured through environment variables:

  CB_API_KEY               API key for all model calls. Per-family overrides are
                           supported as CB_API_KEY_<FAMILY>, where <FAMILY> is the
                           first '-'-separated token of the model id, uppercased
                           (e.g. CB_API_KEY_GPT, CB_API_KEY_CLAUDE, CB_API_KEY_KIMI).
  CB_MESSAGES_URL          Anthropic Messages endpoint
                           (default https://api.anthropic.com/v1/messages)
  CB_CHAT_URL              OpenAI Chat Completions endpoint
                           (default https://api.openai.com/v1/chat/completions)
  CB_RESPONSES_URL         OpenAI Responses endpoint
                           (default https://api.openai.com/v1/responses)
  CB_AGENT_MODEL           coding agent for phase A and both continuation
                           conditions (the paper uses kimi-k3, thinking disabled)
  CB_WORK_ROOT             runtime scratch root for extracted repos, working
                           directories and summary caches (default <repo>/work)
  CB_RESULTS_ROOT          results root (default <repo>/results)
  CB_VENV_ROOT             per-instance virtualenv root (default /tmp/compact_bench_venv)
  CB_S_ROUND               re-seed round for K=3 verification: "2" or "3"
                           (empty = first seed r1)
  CB_SUMMARIZER_MAX_TOKENS compression output cap in tokens (default 16384)
"""
import json
import hashlib
import os
import re
import time
import urllib.request
from collections import defaultdict

MESSAGES_URL = os.environ.get("CB_MESSAGES_URL", "https://api.anthropic.com/v1/messages")
CHAT_URL = os.environ.get("CB_CHAT_URL", "https://api.openai.com/v1/chat/completions")
RESPONSES_URL = os.environ.get("CB_RESPONSES_URL", "https://api.openai.com/v1/responses")
AGENT_MODEL = os.environ.get("CB_AGENT_MODEL", "kimi-k3")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_ROOT = os.environ.get("CB_WORK_ROOT", os.path.join(_REPO_ROOT, "work"))
RESULTS_ROOT = os.environ.get("CB_RESULTS_ROOT", os.path.join(_REPO_ROOT, "results"))
VENV_ROOT = os.environ.get("CB_VENV_ROOT", "/tmp/compact_bench_venv")

S_ROUND = os.environ.get("CB_S_ROUND", "").strip()
S_SFX = f".r{S_ROUND}" if S_ROUND else ""

SUMMARIZER_MAX_TOKENS = int(os.environ.get("CB_SUMMARIZER_MAX_TOKENS", "16384"))
# 16384 rather than 6144: thinking-style models spend output budget on reasoning
# first, and a small cap can leave the visible summary empty.
CONT_MAX_TOKENS = 4096
H_CAP = 250_000


def file_id(value):
    """Keep common model IDs readable, and safely encode provider/model IDs."""
    if re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        return value
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".") or "model"
    return slug + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]


def save_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def key_for_model(model):
    """Resolve the API key for a model: per-family override first, then CB_API_KEY."""
    fam = re.split(r"[-/]", model, maxsplit=1)[0].upper()
    key = os.environ.get(f"CB_API_KEY_{fam}") or os.environ.get("CB_API_KEY")
    if not key:
        raise RuntimeError(
            f"no API key for model {model!r}: set CB_API_KEY or CB_API_KEY_{fam}")
    return key


def post_with_retry(req, tries=15):
    """POST JSON with bounded exponential backoff; retryable on 429/5xx and
    transient network errors."""
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            body = b""
            try:
                body = (e.read() or b"")[:150]
            except Exception:
                pass
            print(f"[llm-retry {i + 1}/{tries}] HTTP {e.code} body={body!r}", flush=True)
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(600, 30 * (2 ** i)))
                continue
            raise
        except Exception as e:
            last = e
            print(f"[llm-retry {i + 1}/{tries}] {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(min(180, 15 * (2 ** i)))
    raise last


def call_messages(model, msgs, key, max_tokens, thinking_disabled=False):
    """Anthropic Messages call. Returns the concatenated text parts."""
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if thinking_disabled:
        # explicit thinking switch is a kimi-family extension; other models
        # reject it with HTTP 400, so only enable per caller request
        body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(
        MESSAGES_URL, data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    d = post_with_retry(req)
    if "content" not in d or not isinstance(d["content"], list):
        raise RuntimeError("Messages endpoint returned an invalid response")
    return "".join(p.get("text", "") for p in d.get("content", []) if p.get("type") == "text")


def call_chat(model, msgs, key, max_tokens):
    """OpenAI Chat Completions call. Returns the message content."""
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    req = urllib.request.Request(
        CHAT_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    d = post_with_retry(req)
    return d["choices"][0]["message"]["content"]


def call_responses(model, user_text, key, max_output_tokens=16384, timeout=240, tries=15):
    """Read a completed JSON response, or SSE from compatible gateways."""
    body = {"model": model, "input": user_text, "max_output_tokens": max_output_tokens,
            "stream": False}
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                RESPONSES_URL, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            if raw.lstrip().startswith("{"):
                data = json.loads(raw)
                if data.get("status") != "completed":
                    raise RuntimeError(f"response status={data.get('status')}")
                return "".join(
                    part.get("text", "")
                    for item in data.get("output", []) if item.get("type") == "message"
                    for part in item.get("content", []) if part.get("type") == "output_text"
                )
            parts, status = [], None
            for line in raw.splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                t = d.get("type")
                if t == "response.output_text.delta":
                    parts.append(d.get("delta", ""))
                elif t in ("response.completed", "response.incomplete", "response.failed"):
                    status = d.get("response", {}).get("status")
                elif t == "error":
                    raise RuntimeError("Responses endpoint returned an error event")
            if status != "completed":
                raise RuntimeError(f"response status={status} (no usable text)")
            return "".join(parts)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(600, 30 * (2 ** i)))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(min(180, 15 * (2 ** i)))
    raise last


def summarize(model, user_text, key, max_tokens=SUMMARIZER_MAX_TOKENS):
    """One compression call with the endpoint routing used by the benchmark:
    gpt-* models go to Responses (reasoning budget), everything else to
    Messages; kimi-family models run with thinking disabled. Successful empty text is returned as an empty summary."""
    protocol = os.environ.get("CB_SUMMARIZER_API", "auto")
    if protocol == "responses" or (protocol == "auto" and model.startswith("gpt")):
        return call_responses(model, user_text, key, max_output_tokens=max_tokens)
    if protocol == "chat":
        text = call_chat(model, [{"role": "user", "content": user_text}], key, max_tokens)
        if not isinstance(text, str):
            raise RuntimeError("Chat endpoint returned no text content")
        return text
    thinking = model.startswith(("kimi", "vendor/kimi", "internal/kimi"))
    return call_messages(model, [{"role": "user", "content": user_text}],
                         key, max_tokens, thinking_disabled=thinking)


# ---------------------------------------------------------------------------
# The five production compaction prompts, transcribed verbatim from the
# respective agent CLIs (see the paper appendix for provenance).
# ---------------------------------------------------------------------------
PROMPTS = {
    "claude": "Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.\nThis summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.\nYour summary should include the following sections:\n1. Primary Request and Intent\n2. Key Technical Concepts\n3. Files and Code Sections (include full code snippets where applicable)\n4. Errors and fixes\n5. Problem Solving\n6. All user messages\n7. Pending Tasks\n8. Current Work\n9. Optional Next Step",
    "codex": "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\nInclude:\n- Current progress and key decisions made\n- Important context, constraints, or user preferences\n- What remains to be done (clear next steps)\n- Any critical data, examples, or references needed to continue\nBe concise, structured, and focused on helping the next LLM seamlessly continue the work.",
    "opencode": "You are an anchored context summarization assistant for coding sessions.\nSummarize only the conversation history you are given. The newest turns may be kept verbatim outside your summary, so focus on the older context that still matters for continuing the work.\nAlways follow the exact output structure requested by the user prompt. Keep every section, preserve exact file paths and identifiers when known, and prefer terse bullets over paragraphs.\nDo not answer the conversation itself. Respond in the same language as the conversation.",
    "gemini": "You are an expert context compressor. Your job is to drastically shorten the following conversation while preserving the absolute core semantic meaning, facts, and intent. Omit all conversational filler, pleasantries, or redundant information. Return ONLY the compressed summary.",
    "kimi": "You are about to run out of context. Write a first-person handoff note to yourself so you can seamlessly continue this task after the earlier conversation is cleared.\n--- This message is a direct task, not part of the above conversation ---\nWrite the note as your own continuing train of thought.",
}


# ---------------------------------------------------------------------------
# Transcript construction (block-level caps, then an overall tail cap).
# ---------------------------------------------------------------------------
def textify(content, limit=4000):
    if isinstance(content, str):
        return content[:limit]
    if not isinstance(content, list):
        return ""
    parts = []
    for p in content:
        t = p.get("type")
        if t == "text":
            parts.append(p.get("text", "")[:limit])
        elif t == "tool_use":
            parts.append(f"[tool_use: {p.get('name')} {json.dumps(p.get('input'), ensure_ascii=False)[:500]}]")
        elif t == "tool_result":
            c = p.get("content")
            c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            parts.append(f"[tool_result: {c[:500]}]")
    return "\n".join(parts)


def transcript(H, cap=H_CAP):
    """E2 serialization: 4000/500 per-block caps, then the overall tail cap."""
    lines = []
    for m in H:
        role = m.get("role", "user")
        t = textify(m.get("content"))
        if t.strip():
            lines.append(f"[{role}] {t}")
    return "\n".join(lines)[-cap:]


def build_transcript(H_msgs, cap=H_CAP):
    """E1 serialization: 2000/200 per-block caps, then the overall tail cap."""
    tr = []
    for m in H_msgs:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            tr.append(f"[{role}] {c[:2000]}")
        elif isinstance(c, list):
            for p in c:
                t = p.get("type")
                if t == "text":
                    tr.append(f"[{role}] {p.get('text', '')[:2000]}")
                elif t == "tool_use":
                    tr.append(f"[{role}] [tool_use: {p.get('name')} {json.dumps(p.get('input'), ensure_ascii=False)[:200]}]")
                elif t == "tool_result":
                    cc = p.get("content")
                    cc = cc if isinstance(cc, str) else json.dumps(cc, ensure_ascii=False)
                    tr.append(f"[{role}] [tool_result: {cc[:200]}]")
    return "\n".join(tr)[-cap:]


# ---------------------------------------------------------------------------
# Atom extraction (deterministic regexes; no LLM).
# ---------------------------------------------------------------------------
RE_PATH = re.compile(r"(?<![\w/.\\:-])((?:[A-Za-z]:[\\/])?(?:[\w@~.-]+[\\/])+[\w@~.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|sql|json|ya?ml|toml|ini|conf|cfg|md|txt|sh|vue|css|scss|html|xml|env))(?![\w/.\\-])")
RE_JUNK_PATH = re.compile(r"(node_modules|site-packages|\.git/|/dist/|/build/|vendor/|__pycache__|\.venv|/tmp/|/proc/)", re.I)
RE_SHA = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{7,40}(?![0-9a-fA-F])")
RE_PR = re.compile(r"(?:PR|MR|pull request|merge request|#)\s*#?(\d{2,6})\b")
RE_ERR = re.compile(r"\b([A-Z][\w]*(?:Error|Exception|Timeout|Refused|Denied|Failed|Overflow|Underflow|Violation|panic|FATAL))\b")
RE_URL = re.compile(r"https?://[^\s)\]\"'>]+")
RE_KV = re.compile(r"\b([A-Z][A-Z0-9_]{2,31})\s*=\s*(\S{1,60})")
RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?\b")


def atoms_in_text(text):
    out = defaultdict(set)
    if not text:
        return out
    text = text.replace("\\n", " ").replace("\n", " ")
    for m in RE_PATH.finditer(text):
        v = m.group(1).replace("\\", "/")
        if RE_JUNK_PATH.search(v):
            continue
        segs = [x for x in v.split("/") if x]
        if len(segs) < 2 or len(v) < 6 or v.endswith("/"):
            continue
        if sum(len(x) for x in segs) / len(segs) < 2:
            continue
        v2 = re.sub(r"(\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|sql|json|ya?ml|toml|ini|conf|cfg|md|txt|sh|vue|css|scss|html|xml|env))/(?=[\w@~.-])", r"\1\n", v)
        for part in v2.split("\n"):
            if part and not part.startswith("/"):
                out["path"].add(part)
    for m in RE_SHA.finditer(text):
        if re.search(r"[a-f]", m.group(0)):
            out["sha"].add(m.group(0))
    for m in RE_PR.finditer(text):
        out["pr"].add(m.group(1))
    for m in RE_ERR.finditer(text):
        out["error"].add(m.group(1))
    for m in RE_URL.finditer(text):
        out["url"].add(m.group(0)[:120])
    for m in RE_KV.finditer(text):
        out["kv"].add(f"{m.group(1)}={m.group(2)}")
    for m in RE_IP.finditer(text):
        out["ip"].add(m.group(0))
    return out


def anchor_baseline(H, cap_chars=400_000, per_type=40):
    """Mechanical anchor upper bound: regex-extract identifiers from H, no LLM."""
    anchor = atoms_in_text(json.dumps(H, ensure_ascii=False)[:cap_chars])
    lines = []
    for k, vs in anchor.items():
        for v in sorted(vs)[:per_type]:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)
