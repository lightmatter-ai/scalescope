"""Pull a fresh snapshot of InferenceX benchmark data into the panel's raw store.

InferenceX (https://inferencex.semianalysis.com/) is SemiAnalysis's
open continuous-inference benchmark. The dashboard is backed by a public
REST API; we hit `/api/v1/benchmarks?model=<display_name>` per model and
archive the JSON arrays to `../benchmark_data/inferencex_raw/`, which is the
corpus consumed by `run_validation_panel.py`.

Run (from anywhere):
    python validation/extraction/inferencex_fetch.py

Output is one `<DisplayName>.json` per model (a JSON list of benchmark
rows). Each row's parallelism fields (`decode_tp`, `decode_ep`,
`decode_dp_attention`, `disagg`, ...) are read verbatim by the panel
loader; this script does not transform them.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import urllib.parse
import urllib.request

# Frontend display names; mapped to one or more DB model keys server-side.
# Only models with a ScaleScope model_zoo spec are exercised by the panel,
# but we archive the full set so the raw corpus is reusable.
MODELS: list[str] = [
    "DeepSeek-R1-0528",
    "gpt-oss-120b",
    "Llama-3.3-70B-Instruct-FP8",
    "Qwen-3.5-397B-A17B",
    "Kimi-K2.5",
    "MiniMax-M2.5",
    "GLM-5",
    "DeepSeek-V4-Pro",
]

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "benchmark_data" / "inferencex_raw"
API_BASE = "https://inferencex.semianalysis.com/api/v1/benchmarks"


def fetch_model(display_name: str) -> bytes:
    """Fetch the full benchmark history for one model as decompressed JSON bytes."""
    url = f"{API_BASE}?model={urllib.parse.quote(display_name)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "ScaleScope-validation-fetcher/1.0",
        },
    )
    raw = urllib.request.urlopen(req, timeout=60).read()
    try:
        return gzip.decompress(raw)
    except (OSError, gzip.BadGzipFile):
        return raw


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for display in MODELS:
        body = fetch_model(display)
        rows = json.loads(body)
        out = OUT_DIR / f"{display.replace('/', '_')}.json"
        out.write_bytes(body)
        print(f"  wrote {out.name:40} rows={len(rows):>5}  size={len(body) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
