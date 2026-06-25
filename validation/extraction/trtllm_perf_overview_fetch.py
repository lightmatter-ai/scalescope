"""Scrape NVIDIA's TensorRT-LLM perf-overview throughput tables into YAML.

Source: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html

The page publishes one HTML table per (model, precision). Each table has a
`GPU:` header row, a `TP Size` row, and `ISL, OSL -> Total Output
Throughput (tokens/sec)` data rows, with one column per GPU variant. This
script parses those tables and emits the schema consumed by
`run_validation_panel.py`:

    models:
      <key>:
        precision: fp8 | fp4
        huggingface_repo: ...
        runs:
          - { gpu: H100, tp: 1, isl: 128, osl: 128, total_output_tps: ... }

Only the models in `KNOWN_MODELS` are emitted (those ScaleScope has a
model_zoo spec for); any other heading found on the page is reported and
skipped. The parser fails fast if a known model's table is missing or
structurally unexpected, so a page-layout change is loud rather than
silent.

Run (writes to stdout by default; redirect or pass --out to persist):
    python validation/extraction/trtllm_perf_overview_fetch.py --out \
        validation/benchmark_data/trtllm_perf_overview.yaml

    # Parse a previously-saved page instead of fetching:
    python validation/extraction/trtllm_perf_overview_fetch.py --html page.html
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import urllib.request
from html.parser import HTMLParser

URL = "https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html"

# Page heading -> emitted YAML block. The key MUST match the top-level
# model id the panel loader (`_TRT_MODEL_TO_SS`) keys on. `alias` is
# documentation only (the loader maps via the key); None omits the field.
KNOWN_MODELS: dict[str, dict] = {
    "Llama 3.1 8B FP8": {
        "key": "llama3_1_8b",
        "precision": "fp8",
        "huggingface_repo": "nvidia/Llama-3.1-8B-Instruct-FP8",
        "alias": None,
    },
    "Llama 3.3 70B FP8": {
        "key": "llama3_3_70b_fp8",
        "precision": "fp8",
        "huggingface_repo": "nvidia/Llama-3.3-70B-Instruct-FP8",
        "alias": "llama3_70b",
    },
    "Llama 3.1 405B FP8": {
        "key": "llama3_1_405b_fp8",
        "precision": "fp8",
        "huggingface_repo": "nvidia/Llama-3.1-405B-Instruct-FP8",
        "alias": "llama3_405b",
    },
    "Llama 3.3 70B FP4": {
        "key": "llama3_3_70b_fp4",
        "precision": "fp4",
        "huggingface_repo": "nvidia/Llama-3.3-70B-Instruct-FP4",
        "alias": "llama3_70b",
    },
    "Llama 3.1 405B FP4": {
        "key": "llama3_1_405b_fp4",
        "precision": "fp4",
        "huggingface_repo": "nvidia/Llama-3.1-405B-Instruct-FP4",
        "alias": "llama3_405b",
    },
}

_ISL_OSL_RE = re.compile(r"^\s*([\d,]+)\s*,\s*([\d,]+)\s*$")


class _TableParser(HTMLParser):
    """Collect (heading, rows) for every `<table>`, tracking the last `<h4>`."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[str, list[list[str]]]] = []
        self._heading: str | None = None
        self._in_h4 = False
        self._h4: list[str] = []
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "h4":
            self._in_h4, self._h4 = True, []
        elif tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "h4":
            self._in_h4 = False
            # The trailing "#"/"¶" comes from the Sphinx headerlink anchor.
            self._heading = "".join(self._h4).strip().rstrip("#\u00b6").strip()
        elif tag == "table" and self._rows is not None:
            self.tables.append((self._heading or "", self._rows))
            self._rows = None
        elif tag == "tr" and self._row is not None:
            self._rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._in_h4:
            self._h4.append(data)
        elif self._cell is not None:
            self._cell.append(data)


def _find_label_row(rows: list[list[str]], label: str) -> list[str]:
    for row in rows:
        if label in row:
            return row
    raise ValueError(f"table is missing a {label!r} row")


def _parse_table(heading: str, rows: list[list[str]]) -> list[dict]:
    """Turn one model table into a list of run dicts."""
    header = _find_label_row(rows, "GPU:")
    tp_row = _find_label_row(rows, "TP Size")
    gpu_label_idx = header.index("GPU:")
    # GPU columns are the non-empty header cells after the "GPU:" label.
    gpu_cols = [j for j in range(gpu_label_idx + 1, len(header)) if header[j]]
    if not gpu_cols:
        raise ValueError(f"{heading!r}: no GPU columns found")

    runs: list[dict] = []
    for row in rows:
        if not row:
            continue
        m = _ISL_OSL_RE.match(row[0])
        if not m:
            continue
        isl = int(m.group(1).replace(",", ""))
        osl = int(m.group(2).replace(",", ""))
        for j in gpu_cols:
            if j >= len(row) or not row[j]:
                continue  # value absent for this GPU/shape on the source page
            tps = float(row[j].replace(",", ""))
            runs.append(
                {
                    "gpu": header[j],
                    "tp": int(tp_row[j].replace(",", "")),
                    "isl": isl,
                    "osl": osl,
                    "total_output_tps": tps,
                }
            )
    if not runs:
        raise ValueError(f"{heading!r}: no ISL/OSL data rows parsed")
    return runs


def scrape(html: str) -> dict[str, list[dict]]:
    """Return {page_heading: [run, ...]} for every KNOWN_MODELS heading."""
    parser = _TableParser()
    parser.feed(html)
    by_heading: dict[str, list[dict]] = {}
    for heading, rows in parser.tables:
        if heading not in KNOWN_MODELS:
            if heading:
                print(f"  skip (not modeled): {heading}", file=sys.stderr)
            continue
        by_heading[heading] = _parse_table(heading, rows)
    missing = [h for h in KNOWN_MODELS if h not in by_heading]
    if missing:
        raise ValueError(
            f"Expected model tables not found on page: {missing}. "
            f"The page layout may have changed; update KNOWN_MODELS."
        )
    return by_heading


def render_yaml(by_heading: dict[str, list[dict]], snapshot_date: str) -> str:
    lines = [
        "# NVIDIA TensorRT-LLM perf-overview throughput tables.",
        "#",
        f"# Source: {URL}",
        f"# Snapshot date: {snapshot_date}. Auto-generated by",
        "# validation/extraction/trtllm_perf_overview_fetch.py.",
        "#",
        "# Metric: `Total Output Throughput (tokens/sec)` — system-wide aggregate",
        "# output token rate at saturating concurrency (offline / infinite-rate",
        "# request stream). Per-user TPOT is NOT published on this page.",
        "",
        "schema_version: 1",
        'source: "NVIDIA TensorRT-LLM perf-overview"',
        f'url: "{URL}"',
        'metric: "Total Output Throughput (tokens/sec) — system aggregate"',
        "",
        "models:",
        "",
    ]
    for heading, spec in KNOWN_MODELS.items():
        runs = by_heading[heading]
        lines.append(f"  {spec['key']}:")
        lines.append(f"    precision: {spec['precision']}")
        lines.append(f"    huggingface_repo: {spec['huggingface_repo']}")
        if spec["alias"] is not None:
            lines.append(f"    aliased_to_scalescope_model: {spec['alias']}")
        lines.append("    runs:")
        for r in runs:
            lines.append(
                f"      - {{ gpu: {r['gpu']}, tp: {r['tp']}, "
                f"isl: {r['isl']}, osl: {r['osl']}, "
                f"total_output_tps: {r['total_output_tps']:.2f} }}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url", default=URL, help="Page URL to fetch.")
    parser.add_argument("--html", default=None, help="Parse a local HTML file instead of fetching.")
    parser.add_argument("--out", default=None, help="Output YAML path (default: stdout).")
    args = parser.parse_args(argv)

    if args.html is not None:
        with open(args.html, encoding="utf-8") as f:
            html = f.read()
        snapshot_date = datetime.date.today().isoformat()
    else:
        req = urllib.request.Request(
            args.url, headers={"User-Agent": "ScaleScope-validation-fetcher/1.0"}
        )
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        snapshot_date = datetime.date.today().isoformat()

    by_heading = scrape(html)
    text = render_yaml(by_heading, snapshot_date)
    total = sum(len(v) for v in by_heading.values())
    print(f"Parsed {len(by_heading)} model tables, {total} run rows.", file=sys.stderr)

    if args.out is None:
        sys.stdout.write(text + "\n")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
