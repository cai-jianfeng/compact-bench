---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - zh
tags:
  - code
  - agents
  - context-compaction
  - benchmark
pretty_name: Compact-Bench
viewer: false
---

# Compact-Bench

Evaluation inputs for [Compact-Bench](https://github.com/cai-jianfeng/compact-bench).
The runner downloads the required files from this repository automatically.

| Track | What it measures | Data |
|---|---|---|
| Task Preservation | Whether a compacted history supports completing a repository repair | 35 tasks, each with a frozen history and repository snapshot |
| Recurring-Detail Retention | Literal retention of details needed again later | 146 sessions (143 real, 3 simulated), with 9,330 annotated atoms |
| Production-Chain Retention | Detail retention across recorded compaction boundaries | 106 real chains and 224 boundaries |

## Run

Clone the code, configure model credentials and endpoints using `.env.example`,
then run:

```bash
python run.py --model YOUR_SUMMARIZER_MODEL
```

The full evaluation needs Linux, Python 3.9+, `git`, and `patch`; a Docker Compose
entry point is provided in the code repository. To run the deterministic
production-chain evaluation without model credentials:

```bash
python run.py --track production
```

No manual dataset download or `datasets` installation is required.
Use `--revision` to select a specific HF commit or release tag.

## Files

```text
e1/instances.json   Task metadata, issue statements, patches and test lists
e1/H3/             35 frozen histories, one JSON file per task
e1/d3snap/         35 matching working-directory snapshots, in tar.gz format
e1/prepass24.txt    Historical cohort identifiers (see below)
e2/sessions.jsonl  Histories and recurring-detail annotations
e3/chains.jsonl    Recorded summaries and following-request annotations
release.json      Public release metadata
SHA256SUMS        File checksums
```

The filenames retain the original internal track IDs. They map to the three
named tracks in the order shown above.

`prepass24.txt` is a historical selection of 24 instances, including the two
excluded pytest instances (`pytest-dev__pytest-10051` and
`pytest-dev__pytest-7236`). The current runner computes the PRE-pass denominator
from the current run and omits these two instances from aggregate task rates.

Session records contain `sample_id`, `client`, history `H`, and `atoms` with
`kind` and `value`. Chain records contain `sess`, `client`, `n_bounds`, and
`per_k`; each boundary includes the recorded summary `S_k` and atom groups
`atomsF`. These files contain nested tool messages with different content types;
use the benchmark's JSON readers. The tabular dataset viewer is disabled.

## Provenance and release processing

Task inputs derive from SWE-bench and the public repositories listed in
`NOTICE.md`; each task records its upstream repository and base commit.
Session and chain records come from coding-agent harness runs, with three
simulated sessions included in the retention track. The five production
compaction prompts are distributed in the code repository.

The public export replaces detected credential strings consistently across
histories, summaries and labels. It retains the task histories and repository
snapshots unchanged. This processing is credential redaction, not a claim of
exhaustive anonymization. The original inputs are kept separately by the
maintainer; release processing is implemented in `scripts/prepare_data.py`.

## License

Compact-Bench's original contributions are provided under Apache-2.0.
Third-party source files and other upstream material retain their applicable
licenses and notices; see `NOTICE.md` and the license files inside the snapshots.
