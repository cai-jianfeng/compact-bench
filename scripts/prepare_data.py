"""Create a separate public data copy; never change the original input directory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

PROVIDER_KEY = re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")
BEARER = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._-]{20,})")
ASSIGNMENT = re.compile(
    r"""(?i)(?:api[_-]?key|access[_-]?token|hf_token|password|secret)[\s"'\\]*[:=][\s"'\\]*([A-Za-z0-9_./+=-]{12,})""")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def redact_records(records):
    """Use one deterministic replacement for each value in every related field."""
    replacements = {}
    kinds = Counter()

    def add(value, kind):
        if not value or value in replacements:
            return
        digest = hashlib.sha256(value.encode()).hexdigest()
        # Alphabetic identifiers avoid introducing accidental numeric matches
        # for the benchmark's literal PR-number atoms.
        alphabetic = digest.translate(str.maketrans("0123456789abcdef", "abcdefghijklmnop"))
        replacement = ("REDACTED_" + alphabetic * (len(value) // 64 + 1))[:len(value)]
        replacements[value] = replacement
        kinds[kind] += 1

    for row in records:
        for text in strings(row):
            for match in PROVIDER_KEY.finditer(text):
                add(match.group(), "credential")
            for pattern in (BEARER, ASSIGNMENT):
                for match in pattern.finditer(text):
                    value = match.group(1)
                    if len(set(value)) >= 8 and not re.search(r"(?i)example|placeholder|your.?key|redacted", value):
                        add(value, "credential")
    if not replacements:
        return records, dict(kinds)
    # Literal replacement also reaches atoms, paths and nested serialized payloads.
    pattern = re.compile("|".join(re.escape(value)
                                 for value in sorted(replacements, key=len, reverse=True)))

    def replace(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: replacements[match.group()], value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value
    return [replace(row) for row in records], dict(kinds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate directories")
    if output.exists():
        parser.error("Choose a new, empty output directory")
    required = ("e1/instances.json", "e2/sessions.jsonl", "e3/chains.jsonl")
    for name in required:
        if not (source / name).is_file():
            parser.error(f"Missing {name}")
    with (source / required[1]).open(encoding="utf-8") as stream:
        sessions = [json.loads(line) for line in stream if line.strip()]
    with (source / required[2]).open(encoding="utf-8") as stream:
        chains = [json.loads(line) for line in stream if line.strip()]
    rows, changes = redact_records(sessions + chains)
    # Public upstream task inputs/snapshots are kept byte-for-byte.
    shutil.copytree(source, output)
    for name, selected in ((required[1], rows[:len(sessions)]), (required[2], rows[len(sessions):])):
        with (output / name).open("w", encoding="utf-8", newline="\n") as destination:
            for row in selected:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    repository = Path(__file__).resolve().parents[1]
    for original, target in (("docs/dataset-card.md", "README.md"),
                             ("docs/dataset-notice.md", "NOTICE.md"),
                             ("LICENSE", "LICENSE")):
        shutil.copyfile(repository / original, output / target)
    summary = {"version": "public-v1", "sessions": len(sessions), "chains": len(chains),
               "replacement_counts": changes,
               "note": "Credential patterns are replaced consistently in histories, summaries and labels. This is not exhaustive anonymization."}
    (output / "release.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(output).as_posix())
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Prepared data: {output}")


if __name__ == "__main__":
    main()
