# Compact-Bench

Evaluate context compaction for coding agents. The runner automatically downloads
its inputs from [Hugging Face](https://huggingface.co/datasets/caijanfeng/compact-bench)
and caches them locally. No manual dataset download is needed.

| Track | Evaluation | Inputs |
|---|---|---|
| Task Preservation | Full-history versus summary-only continuation, judged by repository tests | 35 repository repairs and frozen states |
| Recurring-Detail Retention | Literal detail retention under five production prompts | 146 sessions: 143 real and 3 simulated |
| Production-Chain Retention | Retention across recorded compaction boundaries | 106 chains, 224 boundaries |

## Run

For all three tracks, use Linux and Python 3.9+ (a current patch release) with `git` and `patch` installed,
or use the Docker command below. Python 3.9 is recommended for the historical task
repositories. Retention and production scoring also run on Windows and macOS.
The harness itself uses only Python's standard library; task dependencies are
installed automatically into separate environments.

Set your model credentials and endpoints once, either as environment variables
or in `.env` (copy `.env.example` and fill it in). Then, from the cloned repository:

```bash
python run.py --model YOUR_SUMMARIZER_MODEL
```

This runs all three tracks, downloads their data from HF, creates the task
environments, reuses the PRE baseline, performs the required task verification
rounds, and writes JSON/Markdown reports under `results/<run-id>/`.
Repeat the same command to resume completed cells. Different model/configuration
settings use separate run directories.

Alternatively, with Docker Compose installed and `.env` configured:

```bash
docker compose run --build --rm bench --model YOUR_SUMMARIZER_MODEL
```

Docker builds the Linux environment automatically. Results appear in `results/`;
data and task environments are cached across container runs. A full task run
needs several GB of disk space and model API access.

### Select a track or a small run

```bash
# Full dataset, no model API needed: checks HF download and deterministic scoring.
python run.py --track production

# Two sessions, five compaction prompts per session.
python run.py --track retention --model YOUR_SUMMARIZER_MODEL --limit 2

# Mechanical retention reference, no model API needed.
python run.py --track retention --baseline-only

# One task, using the fixed continuation agent configured below.
python run.py --track task --model YOUR_SUMMARIZER_MODEL --instance pylint-dev__pylint-6903

# Multiple summarizers share the same PRE controls within this run.
python run.py --model MODEL_A MODEL_B
```

`--limit N` limits task instances and retention sessions. `--prompt` selects the
task compaction prompt (`claude`, `codex`, `opencode`, `gemini`, `kimi`); retention
always evaluates all five. Display names are Claude Code, Codex, opencode,
Gemini CLI and Kimi Code. Prompt text is in `harness/cb_common.py`.

## Model configuration

The default transport matches the benchmark gateway: GPT-prefixed summarizers
use Responses, other summarizers use Anthropic-compatible Messages. The
continuation agent uses the Messages tool-call protocol and defaults to `kimi-k3`.
Your gateway must serve the requested model IDs. An API key alone does not make
a model available at a provider's default URL.

| Variable | Purpose |
|---|---|
| `CB_API_KEY` | Model API key; `CB_API_KEY_<FAMILY>` overrides it per model family |
| `CB_AGENT_MODEL` | Fixed continuation agent; default `kimi-k3` |
| `CB_MESSAGES_URL` | Full Messages endpoint, including `/v1/messages` |
| `CB_RESPONSES_URL` | Full Responses endpoint, including `/v1/responses` |
| `CB_CHAT_URL` | Full Chat Completions endpoint |
| `CB_SUMMARIZER_API` | `auto` (default), `messages`, `responses`, or `chat` |
| `CB_PYTHON` | Task environment Python executable; defaults to the runner interpreter |

For a Chat Completions summarizer, set `CB_SUMMARIZER_API=chat` and `CB_CHAT_URL`.
Task continuation still requires a Messages-compatible tool endpoint. The
retention-only track does not need the continuation agent. Kimi-family calls
explicitly disable thinking, matching the benchmark setup.

## Data and results

Data comes from `caijanfeng/compact-bench` by default. Override it with
`--dataset OWNER/DATASET`; select a tag/commit with `--revision REVISION`.
`HF_TOKEN` supports restricted HF repositories. `CB_DATA_MIRROR=/path/to/data`
uses a strictly local copy with the same directory layout. `CB_DATA_CACHE`
changes the download cache directory.

Frozen task histories and repository snapshots are loaded as pairs. Missing
frozen inputs stop that task instead of silently generating a different state.
The original low-level runners also support explicitly requested regeneration
with `CB_REGENERATE_STATE=1`.

The task report separates preserved, damaged, rescued and unsolved cases.
For first-round damage or rescue, the default run adds two verification rounds.
Stable damage requires two POST failures across three rounds. With `--rounds 1`,
verification is intentionally left pending; missing rounds are not scored as zero.
Two known excluded tasks (`pytest-dev__pytest-10051` and `pytest-dev__pytest-7236`)
are omitted from aggregate task rates. PRE-pass denominators are computed from
the current run, rather than assumed to equal the paper's historical denominator.

Useful outputs:

- `production/e3_report.md` and `e3_results.jsonl`
- `retention/e2_report_<model>.md` and `.json`, with per-sample summaries in JSONL
- `task/task_report.md` and `.json`, with PRE/POST records and `errors.json`
- `run.json`: model and data configuration, without API keys

The scorer preserves successful empty summaries as zero retention; transport or
parse errors are reported separately. Re-running retries failed retention cells.

## Development

```bash
python -m unittest discover -s tests -v
```

Tests use local HTTP fixtures and synthetic API responses, without API keys or
paid model calls. Legacy scripts in `scripts/run_e*.sh` forward to `run.py`.

## License

The framework is under Apache-2.0 (`LICENSE`). Upstream repository snapshots,
issue content, and session data retain their own applicable terms; see the HF
dataset card and notices. Model services require the user's own credentials.
