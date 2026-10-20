"""Download benchmark inputs from Hugging Face on first use."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

_ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = os.environ.get("CB_HF_DATASET", "caijanfeng/compact-bench")
DATA_REVISION = os.environ.get("CB_DATA_REVISION", "main")
DATA_BASE = os.environ.get(
    "CB_DATA_BASE",
    f"https://huggingface.co/datasets/{DATA_REPO}/resolve/"
    + urllib.parse.quote(DATA_REVISION, safe=""),
).rstrip("/")
_SOURCE_ID = hashlib.sha256(DATA_BASE.encode()).hexdigest()[:12]
DATA_CACHE = os.environ.get("CB_DATA_CACHE", str(_ROOT / "data" / _SOURCE_ID))


def ensure_data(relpath):
    """Return a cached file; CB_DATA_MIRROR uses a strictly local data copy."""
    relative = PurePosixPath(relpath)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relpath or ":" in relpath:
        raise ValueError(f"Invalid dataset path: {relpath}")
    mirror = os.environ.get("CB_DATA_MIRROR")
    if mirror:
        path = Path(mirror) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing local benchmark input: {path}")
        return str(path)
    local = Path(DATA_CACHE) / relative
    if local.is_file():
        return str(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    url = DATA_BASE + "/" + urllib.parse.quote(relative.as_posix(), safe="/")
    headers = {"User-Agent": "compact-bench"}
    token = os.environ.get("HF_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"
    print(f"[data] downloading {relpath}", file=sys.stderr, flush=True)
    for attempt in range(3):
        temporary = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                with tempfile.NamedTemporaryFile(dir=local.parent, delete=False) as output:
                    temporary = output.name
                    shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
                if expected is not None and os.path.getsize(temporary) != int(expected):
                    raise OSError("Incomplete dataset download")
            os.replace(temporary, local)
            return str(local)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise RuntimeError(
                    f"Cannot download {relpath} (HTTP {exc.code}). Check CB_HF_DATASET, "
                    "CB_DATA_REVISION and, for restricted datasets, HF_TOKEN."
                ) from exc
            if attempt == 2:
                raise
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        time.sleep(2 ** attempt)
