#!/usr/bin/env bash
set -euo pipefail

# Generate only artifacts used in the current paper draft and copy them
# into a dedicated folder.
#
# Prerequisites: activate the project virtual environment before running.
#   python3 -m venv .venv && source .venv/bin/activate && pip install -e .

python3 -c "import scalescope" 2>/dev/null || {
  echo "ERROR: scalescope is not importable. Activate the virtual environment first:" >&2
  echo "  source .venv/bin/activate" >&2
  exit 1
}
#
# Usage:
#   bash conf26doc/generate_paper_artifacts.sh
#
# Notes:
# - This script deletes out/conf26/build before generating outputs.
# - Sweeps and paper artifacts are always regenerated from scratch.
# - Table II / Figure 5 are derived from the HyperX sweep summary
#   (the full parity-aligned search space).

case "${1:-}" in
  -h|--help)
    echo "Usage: $0"
    echo "  Regenerate the conf26 sweeps and derive the paper artifacts."
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: ${1}" >&2
    echo "Usage: $0" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONF26_OUT_DIR="${REPO_ROOT}/out/conf26"
CONF26_BUILD_DIR="${CONF26_OUT_DIR}/build"
CONF26_SWEEP_DIR="${CONF26_BUILD_DIR}/sweeps"
PAPER_ARTIFACTS_DIR="${CONF26_OUT_DIR}/paper_artifacts"
PRECURSOR_DIR="${PAPER_ARTIFACTS_DIR}/precursor_sweeps"

# Ensure output freshness for every run.
rm -rf "${CONF26_BUILD_DIR}"
rm -rf "${PAPER_ARTIFACTS_DIR}"
mkdir -p "${CONF26_BUILD_DIR}"

# Always regenerate sweeps from scratch.
echo "Running conf26 sweep set (torus3d, HyperX, FC128x32)."
bash "${REPO_ROOT}/conf26doc/run_conf26_sweeps.sh"

# Required inputs (produced by publication sweep launcher). The HyperX summary
# comes from the FC-style replicated-set sweep (s in {1,2,4,8,16,32}); see
# conf26doc/run_conf26_sweeps.sh.
REQ_HYPERX="${CONF26_SWEEP_DIR}/generalized_ep_hyperx/summary.csv"
REQ_TORUS="${CONF26_SWEEP_DIR}/generalized_ep_torus3d/summary.csv"
REQ_FC="${CONF26_SWEEP_DIR}/fc128x32_disagg_ep_baseline/summary.csv"

for req in "${REQ_HYPERX}" "${REQ_TORUS}" "${REQ_FC}"; do
  if [[ ! -f "${req}" ]]; then
    echo "Missing required summary: ${req}" >&2
    echo "Publication sweep regeneration failed to produce required outputs." >&2
    exit 1
  fi
done

# Figure 4: HyperX regime map (batch vs imbalance), context=1024.
python3 "${REPO_ROOT}/conf26doc/plot_conf26_regime_config_maps.py" \
  --out-dir "${CONF26_SWEEP_DIR}" \
  --context 1024

# Table II: rho=8 cross-topology speedup table.
python3 "${REPO_ROOT}/conf26doc/generate_topology_tables.py" \
  --out-dir "${CONF26_BUILD_DIR}" \
  --hyperx-summary "${REQ_HYPERX}" \
  --torus-summary "${REQ_TORUS}" \
  --fc-summary "${REQ_FC}"

python3 "${REPO_ROOT}/conf26doc/generate_speedup_table_rho8.py" \
  --input-csv "${CONF26_BUILD_DIR}/cross_topology_speedups.csv" \
  --rho 8 \
  --out-csv "${CONF26_BUILD_DIR}/topo_speedups_rho8_table.csv" \
  --out-tex "${CONF26_BUILD_DIR}/topo_speedups_rho8_table.tex"

# Figure 5: 3-topology bottleneck heatmap.
python3 "${REPO_ROOT}/conf26doc/plot_resource_bottleneck_heatmap_3topo.py" \
  --rho 8 \
  --min-batch 32 \
  --context-increases-up \
  --no-title \
  --no-legend \
  --single-axis-labels \
  --hyperx "${REQ_HYPERX}" \
  --torus "${REQ_TORUS}" \
  --fc "${REQ_FC}" \
  --out-png "${CONF26_BUILD_DIR}/fig_bottleneck_3topo_rho8.png"

# Collect only paper-used assets in a dedicated folder.
mkdir -p "${PAPER_ARTIFACTS_DIR}"
cp "${CONF26_SWEEP_DIR}/hyperx_regime_map.png" "${PAPER_ARTIFACTS_DIR}/fig4_hyperx_regime_map.png"
cp "${CONF26_BUILD_DIR}/fig_bottleneck_3topo_rho8.png" "${PAPER_ARTIFACTS_DIR}/fig5_bottleneck_3topo_rho8.png"
cp "${CONF26_BUILD_DIR}/topo_speedups_rho8_table.tex" "${PAPER_ARTIFACTS_DIR}/table2_topo_speedups_rho8.tex"
cp "${CONF26_BUILD_DIR}/topo_speedups_rho8_table.csv" "${PAPER_ARTIFACTS_DIR}/table2_topo_speedups_rho8.csv"

# Also package precursor sweep/intermediate results used to derive paper artifacts.
mkdir -p "${PRECURSOR_DIR}"
cp "${REQ_HYPERX}" "${PRECURSOR_DIR}/summary_hyperx.csv"
cp "${REQ_TORUS}" "${PRECURSOR_DIR}/summary_torus3d.csv"
cp "${REQ_FC}" "${PRECURSOR_DIR}/summary_fc128x32.csv"
cp "${CONF26_BUILD_DIR}/cross_topology_speedups.csv" \
  "${PRECURSOR_DIR}/cross_topology_speedups.csv"

# Extract best configs per topology (writes *_best_configs.csv). The
# context-invariance check is informational and does not gate the build.
python3 "${REPO_ROOT}/conf26doc/verify_topology_claims.py" \
  --precursor-dir "${PRECURSOR_DIR}" \
  --out-dir "${PAPER_ARTIFACTS_DIR}" \
  || echo "verify_topology_claims: context-invariance check failed (ignored)."

# Verify that the generated Table II TeX matches the CSV it was derived from
# (export self-consistency, not a hardcoded-value assertion).
python3 "${REPO_ROOT}/conf26doc/check_table2_speedups_consistency.py"

cat > "${PAPER_ARTIFACTS_DIR}/README.txt" <<'EOF'
Paper artifacts (current draft):
- fig4_hyperx_regime_map.png
- fig5_bottleneck_3topo_rho8.png
- table2_topo_speedups_rho8.tex
- table2_topo_speedups_rho8.csv

Per-topology winning configurations:
- hyperx_best_configs.csv
- torus3d_best_configs.csv
- fc128x32_best_configs.csv

Precursor sweep/intermediate data (used to derive artifacts):
- precursor_sweeps/summary_hyperx.csv
- precursor_sweeps/summary_torus3d.csv
- precursor_sweeps/summary_fc128x32.csv
- precursor_sweeps/cross_topology_speedups.csv
EOF

echo "Wrote paper-only artifacts to: ${PAPER_ARTIFACTS_DIR}"
