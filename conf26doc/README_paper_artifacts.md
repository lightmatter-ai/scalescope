# Paper-Only Artifact Generation

This guide generates only the artifacts used in the current paper draft and stores them in a dedicated folder.

## Script

- `conf26doc/generate_paper_artifacts.sh`

## Output folder

- final artifacts package: `out/conf26/paper_artifacts/`
- sweep/intermediate outputs: `out/conf26/build/sweeps/` and `out/conf26/build/`

## Artifacts produced

- `fig4_hyperx_regime_map.png`
- `fig5_bottleneck_3topo_rho8.png`
- `table2_topo_speedups_rho8.tex`
- `table2_topo_speedups_rho8.csv`
- `hyperx_best_configs.csv`
- `torus3d_best_configs.csv`
- `fc128x32_best_configs.csv`

## Precursor sweep/intermediate results packaged

- `precursor_sweeps/summary_hyperx.csv`
- `precursor_sweeps/summary_torus3d.csv`
- `precursor_sweeps/summary_fc128x32.csv`
- `precursor_sweeps/cross_topology_speedups.csv`

## Usage

### Regenerate everything from scratch (fresh by construction)

```bash
bash conf26doc/generate_paper_artifacts.sh
```

This script removes `out/conf26/build/` and `out/conf26/paper_artifacts/` first, always reruns the full sweep set
(the HyperX sweep spans `s in {1,2,4,8,16,32}`, covering the complete parity-aligned search space),
derives Table II / Figure 5 from that summary, regenerates all paper artifacts, and then runs two steps:

- `verify_topology_claims.py`: extracts the per-topology winning-config CSVs and reports a structural
  context-invariance check (informational; does not gate the build)
- `check_table2_speedups_consistency.py`: confirms internal consistency between the generated Table II LaTeX and the sweep CSV it was derived from
