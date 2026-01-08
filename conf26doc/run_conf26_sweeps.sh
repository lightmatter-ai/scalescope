#!/usr/bin/env bash
set -euo pipefail

# Publication expert-placement sweep launcher.
# Default output root:
#   out/conf26/build/sweeps

HW_LIB="hardware_specs/hardware_library.yaml"
LOGICAL="conf26/topology_zoo/logical/logical_4096_tp2d_32x128_multidim.yaml"
WORKLOAD="conf26/inference_decode_specs/inference_decode_spec__conf26_decode.yaml"
MODEL="conf26/model_zoo/gpt_moe_config__1024_experts__gqa_mlp025.yaml"
OUT_DIR="out/conf26/build/sweeps"

TORUS3D_LOGICAL="conf26/topology_zoo/logical/logical_4096_tp2d_16x16x16_multidim.yaml"
TORUS3D_WORKLOAD="conf26/inference_decode_specs/inference_decode_spec__conf26_decode__torus3d.yaml"

# Torus generalized (3D torus modeled as 16x16x16 logical decomposition)
MPLCONFIGDIR=tmp/mpl_torus scalescope-cli \
  --hardware-lib "$HW_LIB" \
  --physical-topology "conf26/topology_zoo/physical/physical_4096_torus_16x16x16.yaml" \
  --logical-topology "$TORUS3D_LOGICAL" \
  --workload-spec "$TORUS3D_WORKLOAD" \
  --model-spec "$MODEL" \
  --sweep-config "conf26/sweep_configs/expert_placement_generalized_torus_3d.yaml" \
  --pass-mode inference_decode \
  --quiet \
  --output-dir "$OUT_DIR" \
  --run-name "generalized_ep_torus3d" \
  --no-timestamp --overwrite --raw-output

# HyperX generalized
MPLCONFIGDIR=tmp/mpl_hyperx scalescope-cli \
  --hardware-lib "$HW_LIB" \
  --physical-topology "conf26/topology_zoo/physical/physical_4096_hyperx_32x128.yaml" \
  --logical-topology "$LOGICAL" \
  --workload-spec "$WORKLOAD" \
  --model-spec "$MODEL" \
  --sweep-config "conf26/sweep_configs/expert_placement_generalized_fixed_tp.yaml" \
  --pass-mode inference_decode \
  --quiet \
  --output-dir "$OUT_DIR" \
  --run-name "generalized_ep_hyperx" \
  --no-timestamp --overwrite --raw-output

# FC disaggregated constrained baseline
MPLCONFIGDIR=tmp/mpl_fc scalescope-cli \
  --hardware-lib "$HW_LIB" \
  --physical-topology "conf26/topology_zoo/physical/physical_4096_fc128_x32_groups.yaml" \
  --logical-topology "$LOGICAL" \
  --workload-spec "$WORKLOAD" \
  --model-spec "$MODEL" \
  --sweep-config "conf26/sweep_configs/expert_placement_fc128x32_disaggregated.yaml" \
  --pass-mode inference_decode \
  --quiet \
  --output-dir "$OUT_DIR" \
  --run-name "fc128x32_disagg_ep_baseline" \
  --no-timestamp --overwrite --raw-output
