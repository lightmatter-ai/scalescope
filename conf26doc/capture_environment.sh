#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/conf26doc/env}"

mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found. Set PYTHON_BIN=<path-to-python> and retry." >&2
  exit 1
fi

ENV_SUMMARY="${OUT_DIR}/environment_summary.txt"
PIP_FREEZE="${OUT_DIR}/requirements_freeze.txt"

{
  # echo "timestamp_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python_version: $("${PYTHON_BIN}" --version 2>&1)"
  echo "pip_version: $("${PYTHON_BIN}" -m pip --version 2>&1 | awk '{print $2}')"
  if command -v uname >/dev/null 2>&1; then
    echo "uname: $(uname -srm)"
  fi
  if command -v sw_vers >/dev/null 2>&1; then
    echo "sw_vers_product_name: $(sw_vers -productName)"
    echo "sw_vers_product_version: $(sw_vers -productVersion)"
    echo "sw_vers_build_version: $(sw_vers -buildVersion)"
  fi
  # if command -v git >/dev/null 2>&1; then
  #   echo "git_head_sha: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unavailable")"
  # fi
} > "${ENV_SUMMARY}"

"${PYTHON_BIN}" -m pip freeze | grep -vi "scalescope" > "${PIP_FREEZE}"

echo "Wrote environment capture to: ${OUT_DIR}"
echo "  - ${ENV_SUMMARY}"
echo "  - ${PIP_FREEZE}"
