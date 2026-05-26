# ScaleScope - Analytical Modeling Tool

This project provides a tool to analyze and estimate the performance of large language models based on specified hardware configurations,
model architectures, and network topologies.

## Overview

The script takes several configuration files in YAML format as input:

*   **Hardware Library:** Defines basic hardware components (Compute, Memory, Interconnect) and combines them into named `NetworkNode` definitions.
*   **Model Specification:** Describes the transformer model architecture (e.g., hidden size, number of layers, attention heads, MoE parameters).
*   **Workload Specification:** Defines workload parameters like global batch size, sequence length, and parallelism strategies (TP, DP, PP sizes). It is used for training, inference, and other workloads.
*   **Topology Specifications:** Separate files describing the network topology used for Data Parallelism (DP) and Tensor/Pipeline Parallelism (TP/PP) communication.
*   **Sweep Configuration:** Describes which values in the specifications above should be swept over.

Based on these inputs, the tool analyzes performance characteristics.

This repository accompanies the paper “Accelerating Frontier MoE Training with 3D Integrated Optics” (Mikhail Bernadskiy, Peter Carson, Thomas Graham, Taylor Groves, Ho John Lee, Eric Yeh), published in Hot Interconnects 2025, and provides ready-to-run scenarios; see “Hot Interconnects 2025 configurations” below. The paper is available at [arXiv](https://arxiv.org/abs/2510.15893).

## Prerequisites

*   Python 3.11+

## Installation

From the top level `scalescope` directory, install the package using:
```bash
pip install -e .
```

##  Usage

Run the main analysis script from your terminal, providing paths to all required configuration files and specifying the hardware nodes to consider.

```bash
scalescope-cli \
    --hardware-lib <path/to/hardware_library.yaml> \
    --physical-topology <path/to/physical_topology.yaml> \
    --logical-topology <path/to/logical_topology.yaml> \
    --workload-spec <path/to/workload_spec.yaml> \
    --model-spec <path/to/model_spec.yaml> \
    --sweep-config <path/to/sweep_config.yaml> \
    --pass-mode <training|inference_prefill|inference_decode> \
    [--output-file <path/to/output_results.yaml>] \
    [--output-dir <path/to/output_dir> --run-name <name> [--no-timestamp] [--overwrite]] \
    [--raw-output] # Optional
```

### Arguments

* --hardware-lib (Required): Path to the hardware library YAML file.
* --physical-topology (Required): Path to the physical topology YAML file.
* --logical-topology (Required): Path to the logical topology YAML file.
* --workload-spec (Required): Path to the workload specification YAML file.
* --model-spec (Required): Path to the model specification YAML file.
* --sweep-config (Required): Path to the sweep configuration YAML file.
* --pass-mode (Optional): Mode of analysis. Defaults to `training`. Choices:
  - `training` (full forward+backward)
  - `inference_prefill` (prompt processing)
  - `inference_decode` (token generation)
  Note: ensure `--workload-spec` points to the matching spec directory for the selected mode (e.g., `train_specs/...`, `inference_prefill_specs/...`, or `inference_decode_specs/...`).
* --output-file (Optional): Path to save the detailed sweep results as a YAML file. If not provided, a summary is printed to the console.
* --output-dir (Optional): Directory to write sweep artifacts. ScaleScope will create a subdirectory (named by `--run-name`, optionally with a timestamp suffix) containing:
  - `manifest.yaml`: provenance (including git SHA when available) and an index of run IDs to per-run files
  - `runs/<run_id>.yaml`: one detailed structured YAML output per sweep point
* --run-name (Optional): Name for the run subdirectory under `--output-dir`.
* --no-timestamp (Optional): When set, do not append a timestamp to the run subdirectory name (useful for tests/CI).
* --overwrite (Optional): When set, allow overwriting an existing run subdirectory.
* --raw-output (Optional): If specified, disables human-readable formatting for numbers (e.g., bytes, FLOPs, time) in the YAML output, printing raw numeric values instead.

### Config Files

ScaleScope consumes several YAML configuration files. This section summarizes their purpose; see “Configuration reference” below for the complete field lists, and the “Hot Interconnects 2025 configurations” section below for concrete examples.

1. **Hardware library (`--hardware-lib`)**
   - Declares reusable hardware components (compute, memory, interconnect) and assembles them into named node types referenced by other configs.

2. **Model spec (`--model-spec`)**
   - Describes the transformer architecture (hidden size, layers, heads, MLP ratio, vocab size, optional GQA, and MoE-related counts).

3. **Workload spec (`--workload-spec`)**
   - Defines runtime, batching, and parallelism settings for training or inference (batch sizes, sequence lengths, TP/DP/PP variants and sizes), along with memory-tier mappings and overlap efficiencies. Use the directory matching your mode: `train_specs/`, `inference_prefill_specs/`, or `inference_decode_specs/`.

4. **Physical topology (`--physical-topology`)**
   - Describes the physical cluster: which node type is used and how physical devices are connected via specific interconnects across dimensions (e.g., intra-/inter-pod fabrics). Each dimension names the interconnect it uses and its size. Logical topologies reference these dimension names when forming communication groups. See “Configuration reference” for the full field list.

5. **Logical topology (`--logical-topology`)**
   - Defines the logical dimensions used by distributed training/inference (e.g., TP_Attn, TP_MoE, DP_Attn, DP_MoE, EP_*, PP) and maps each to a physical dimension to form collective groups. The `size` on each mapping sets that dimension’s degree on the chosen fabric. When multiple logical mappings share the same interconnect, effective bandwidth is adjusted to account for sharing. See “Configuration reference” for the detailed field list.

6. **Sweep configuration (`--sweep-config`)**
   - Provides parameter sweeps using the **Sweep DSL** described below. It defines `leaves` (aligned multi-parameter case lists or generators) and a `sweep` expression composed with `cartesian` / `zip` / `union`, optionally filtered by `where` verifiers.

## Configuration reference

### Training spec (train_specs/*.yaml)
- **global_batch_size**: Global batch size across all DP ranks.
- **seq_length**: Sequence length used during training.

- **micro_batch_size**: Per-micro-batch size per DP rank.
- **total_train_tokens**: Total number of tokens to train over (drives total runtime).
- **float_size_bytes**: Compute dtype size in bytes (e.g., 2 for bf16).
- **full_precision_param_float_size**: Parameter/optimizer state dtype size in bytes (e.g., 4).
- **optimizer**: Optimizer name (currently only `adam` is supported).
- **memory_device_map_traffic**: Map each component to a memory tier for traffic accounting. Tiers must match the node's `device_memory_tiers` declared in the hardware library (for example, `xpu_memory`, `host_memory`, `remote_memory`, `nvme`). The chosen tier determines which bandwidth is used when converting bytes to time for that component. Currently, configurations use `xpu_memory` for all components; this mapping exists to support future modeling of remote tiers and offloading (e.g., `kv_cache` offload to `remote_memory` during decode).
- **memory_device_map_footprint**: Map each component to a memory tier for footprint/capacity accounting. Uses the same tier namespace as above. Keys: `weights`, `optimizer_states`, `saved_activations`, `gradient_buffers`, `transient_forward_peak`, `transient_backward_peak`. The assigned tier’s capacity is checked when verifying feasibility; changing tiers lets you model sharding or offload (e.g., placing `saved_activations` on `host_memory` to simulate activation offload).
- **comp_comm_efficiency / comp_mem_efficiency / comm_mem_efficiency**: Overlap efficiencies between compute/comm/memory (0.0–1.0).
- **tp_attn_spec.variant / tp_moe_spec.variant**: Tensor parallel variants.
- **dp_spec.variant**: Data parallel variant (only `DDP` is currently supported).
- **pp_spec.variant**: Pipeline parallel variant (only `GPIPE` is currently supported). PP stage count is derived from the logical topology (`PP` mapping size).
- **moe_spec_v2**:
  - **num_colocated_experts**: Experts colocated per XPU.
  - **shared_expert_copies**: Number of shared-expert replicas (or null).
  - **expert_imbalance**: Factor to model skew in token routing across regular experts (≥1.0). Applied as a multiplier to per-expert token volume in both compute and memory models. `1.0` means perfectly balanced routing; e.g., `1.5` assumes the busiest expert processes up to 50% more tokens than the average.

### Inference prefill spec (inference_prefill_specs/*.yaml)
- Same structure as training, typically without optimizer-specific fields:
  - **global_batch_size**, **micro_batch_size**, **seq_length**, **float_size_bytes**
  - **memory_device_map_traffic**, **memory_device_map_footprint**
  - **comp_comm_efficiency**, **comp_mem_efficiency**, **comm_mem_efficiency**
  - **tp_attn_spec.variant**, **tp_moe_spec.variant**, **dp_spec.variant**,  **pp_spec.variant**
  - **moe_spec_v2**: `num_colocated_experts`, `shared_expert_copies`, `expert_imbalance`

### Inference decode spec (inference_decode_specs/*.yaml)
- Fields specific to decoding:
  - **prefill_seq_len**: Sequence length already in KV cache before decoding starts.
  - **num_generated_tokens**: Number of tokens to generate.
- Other fields mirror prefill:
  - **global_batch_size**, **micro_batch_size**, **float_size_bytes**
  - **memory_device_map_traffic** (often `kv_cache: remote_memory`), **memory_device_map_footprint**
  - **comp_comm_efficiency**, **comp_mem_efficiency**, **comm_mem_efficiency**
  - **tp_attn_spec.variant**, **tp_moe_spec.variant**, **dp_spec.variant**, **pp_spec.variant**
  - **moe_spec_v2**: `num_colocated_experts`, `shared_expert_copies`, `expert_imbalance`

## Inference unit sweeps (TP-only, DP=PP=1)

For inference decode analysis without scale-out complications, see the `inference_decode_2026` project files:

- **Physical topology**: `topology_zoo/physical/inference_decode_2026/inference_unit_physical.yaml`
- **Logical topology**: `topology_zoo/logical/inference_decode_2026/inference_unit_logical.yaml`
- **Workload spec**: `inference_decode_specs/inference_decode_2026/inference_decode_spec__inference_unit.yaml`
- **Sweep**: `sweep_configs/inference_decode_2026/attn_proj_decode_inference_unit_tp_sweep.yaml`

This setup models a single *inference unit* whose size is the TP group size, and it sweeps
TP1D vs TP2D while holding DP and PP at 1. This helps isolate how TP strategy and TP size affect
compute, memory traffic, and communication in a single decode step.

### Model spec (model_zoo/*.yaml)
- **d_model**: Hidden dimension.
- **n_layers**: Total number of transformer layers.
- **num_heads**: Attention heads.
- **num_kv_heads**: KV heads (GQA). Defaults to `num_heads` if omitted.
- **mlp_ratio**: FFN hidden size ratio relative to `d_model`.
- **max_sequence_length**: Maximum supported sequence length.
- **vocab_size**: Vocabulary size.
- MoE parameters (supported by the model configuration):
  - **num_regular_experts**: Experts per layer (0 or 1 behaves as dense).
  - **num_selected_experts**: Experts selected per token.

### Physical topology spec (topology_zoo/physical/*.yaml)
- **name**: Topology name.
- **node_definition_ref**: Reference to a hardware node type defined in the hardware library.
- **dimensions[]**: List of physical dimensions with:
  - **name**: Dimension name (e.g., `InterPodFabric`, `IntraPodFabric`).
  - **interconnect_name**: Interconnect key present on the node.
  - **topology_type**: e.g., `switched`, `mesh`, `ring`, etc.
  - **size**: Number of nodes along this dimension.
- Optional:
  - (removed) Expert high-bandwidth group size is inferred from the physical dimension mapped by logical tag `EP_SU`.

### Logical topology spec (topology_zoo/logical/*.yaml)
- **name**: Logical topology name.
- **dimensions[]**: Map logical parallelism to physical dimensions:
  - **logical_tag**: Tag name (e.g., `PP`, `TP_Attn`, `TP_MoE`, `DP_Attn`, `DP_MoE`, `EP_SU`, `EP_SO`).
  - **maps_to_physical_dimension_name**: Physical dimension name from the physical topology.
  - **size**: Degree for that logical dimension.

## Sweep DSL (`--sweep-config`)

ScaleScope uses a small YAML DSL to define sweeps. The sweep file has two main sections:

- **`leaves:`**: named leaf sweeps. Each leaf sets one or more *paths* and enumerates aligned *cases*.
  - A leaf must specify **exactly one** of:
    - `cases`: list of aligned tuples (each tuple length must equal `len(paths)`), or
    - `generator`: `{name: ..., args: {...}}` producing a `list[list[Any]]` aligned to `paths`.
  - Optional: `labels`: a parallel list of label dictionaries aligned 1:1 with `cases`. Labels are metadata only.
- **`sweep:`**: an expression tree over leaves:
  - `ref`: reference a leaf by name
  - `cartesian`: cross product of children
  - `zip`: index-aligned merge of children (**strict by default**; lengths must match)
  - `union`: concatenate child streams
  - Any operator node can include:
    - `where: [verifier_name, ...]` (AND semantics)
    - `on_fail: skip|error` (default: `skip`)

### Path syntax

Paths are written as `spec_key.attr.path`, where `spec_key` is one of the loaded base specs (e.g. `workload_spec`, `model_spec`, `logical_topology`, `physical_topology`).

Special case for logical topology degrees:

- `logical_topology.tag_size.<TAG>` updates the `size` of the logical dimension whose `logical_tag` matches `<TAG>`.

### Empty sweep file

An **empty** sweep YAML (or a YAML file containing only comments) means “no sweep”: ScaleScope runs a **single** evaluation using the base input specs.

### Labels

If a leaf defines `labels`, each run output will include a merged `labels` dictionary (and flattened `label.<key>` columns).

You can also use a label-only leaf (identity leaf) to annotate sweep points without changing any spec values:

```yaml
leaves:
  fabric_label:
    paths: []
    cases: [[]]
    labels:
      - {fabric: electrical}
```

### Report (summary columns + BEST markers)

If `report:` is present in the sweep YAML and you run with `--output-dir`, ScaleScope will write `summary.csv`
into the run directory and record it in `manifest.yaml`.

`report.columns` are extracted by **arbitrary dotted paths** into the per-run YAML record (e.g. `time.total_time_seconds`,
`configuration.tp_xpu_count`, `labels.fabric`, `sweep_case.model_spec.num_regular_experts`).

Example:

```yaml
report:
  columns:
    - {name: fabric, path: labels.fabric}
    - {name: experts, path: sweep_case.model_spec.num_regular_experts}
    - {name: batch, path: configuration.batch_size_per_dp}
    - {name: overall_s, path: time.total_time_seconds}

  best:
    - name: BEST
      group_by: [fabric, experts, batch]
      metric: overall_s
      mode: min

    # Implicit subset example: BEST1D is the subset where N2 == 1
    - name: BEST1D
      group_by: [fabric, batch]
      metric: overall_s
      mode: min
      where:
        - {col: N2, op: "==", value: 1}
```

Notes:
- `best[].where` predicates refer to extracted **column names** (not paths).
- Supported `op`: `==`, `!=`, `<`, `<=`, `>`, `>=`.

### Built-in generators

The sweep DSL supports these generator names:

- `power_of_two` (args: `min_val`, `max_val`) – emits `[[N]]` for powers of two in range
- `perfect_square_factors` (args: `min_val`, `max_val`) – emits `[[n, n]]` for square grids
- `decompose_with_balanced_multiplicands` (args: `min_val`, `max_val`, `step`) – emits `[[x, y]]` balanced factor pairs for each integer in the range

### Built-in verifiers

- `pydantic_valid`: keeps only cases that can be applied to the base specs without Pydantic validation errors.
- `projection_groups_divides_proj_yz`: for decode TP2D sweeps, keeps only cases where `workload_spec.projection_groups`
  divides the resolved projection TP2D YZ size (from `workload_spec.tp_proj_spec.split_2d.yz_tags` and `logical_topology` tag sizes).

### Example sweeps (report + BEST markers)

Sweep configs can include `report:` to produce a compact `summary.csv` with BEST markers.

Example (inference decode, TP-only inference unit sweep):

```bash
scalescope-cli \
  --hardware-lib hardware_specs/hardware_library.yaml \
  --physical-topology topology_zoo/physical/inference_decode_2026/inference_unit_physical.yaml \
  --logical-topology topology_zoo/logical/inference_decode_2026/inference_unit_logical.yaml \
  --workload-spec inference_decode_specs/inference_decode_2026/inference_decode_spec__inference_unit.yaml \
  --model-spec model_zoo/tp2d/gpt_moe_config__1024_experts__gqa.yaml \
  --sweep-config sweep_configs/inference_decode_2026/attn_proj_decode_inference_unit_tp_sweep.yaml \
  --pass-mode inference_decode \
  --output-dir /tmp/scalescope_out --run-name inference_unit_tp_sweep --no-timestamp --overwrite
```

### Example

This sweeps batch size and sequence length as a cartesian product (use a training or prefill workload spec that has `seq_length`):

```yaml
leaves:
  batch:
    paths: [workload_spec.global_batch_size]
    cases:
      - [1]
      - [64]
      - [512]

  seq_len:
    paths: [workload_spec.seq_length]
    cases:
      - [1024]
      - [4096]
      - [8192]

sweep:
  cartesian:
    - ref: batch
    - ref: seq_len
  where: [pydantic_valid]
  on_fail: skip
```

See also `examples/sweep_dsl_example.yaml`.

### Hardware library (hardware_specs/hardware_library.yaml)
- Components:
  - **compute**: Entries with `fp8_nominal`, `bf16_nominal`, `fp32_nominal`, `efficiency`.
  - **memory**: Entries with `bandwidth_per_stack_nominal`, `size_per_stack`, `num_stack`, `efficiency`.
  - **interconnect**: Entries with `type`, `bandwidth_nominal`, `efficiency`, optional `latency_sec`, `num_links`.
- Nodes:
  - **device_compute**: Reference to a compute component.
  - **device_memory_tiers**: Mapping of tier name→memory component reference.
  - **device_interconnects**: Mapping of local interconnect name→interconnect component reference.

### YAML output structure

When `--output-file` is provided, ScaleScope writes a YAML list where each item corresponds to one evaluated configuration (sweep point). Each item contains:

- **configuration**: Top-level run settings (e.g., node name, TP/DP sizes, layers per stage, seq_len, etc.).
- **model_configuration**: Core model parameters (e.g., expert counts, mlp_ratio).
- **expert_placement**: Derived MoE placement stats (e.g., experts per DP rank, complete expert sets).
- **time**: Key timing metrics, including `total_time` (human-readable unless `--raw-output`) and `total_time_seconds`.
- **compute_per_batch_per_xpu**: Forward and backward compute time and FLOPs, plus breakdowns for attention, MoE/FFN, embedding, and unembedding.
- **memory_traffic_per_batch_per_xpu**: Forward and backward memory time and bytes, with component breakdowns.
- **communication_per_batch_per_xpu**: Forward and backward communication time and bytes, including TP and DP communication and MoE all-to-all.
- **ratios**: Helpful aggregate ratios (e.g., comm_to_compute, memory_to_compute).
- **memory_footprint**: Per-tier footprint (weights, optimizer states, saved activations, transients) and `total_peak_memory_per_chip_bytes`.
- **memory_traffic_details_per_batch**: Fine-grained bytes for attention and MoE/FFN per direction, including detailed FFN backward terms.

Note: Use `--raw-output` to keep numeric values unformatted (seconds/bytes/FLOPs) if you plan to post-process the YAML.

### Hot Interconnects 2025 configurations

#### Config 1 (32 Experts)

##### Alternative
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_1__32_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_alternative_train_32e.yaml
```

##### Passage
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/passage_logical_config_1__32_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_passage_train_32e.yaml
```

#### Config 2 (64 Experts)

##### Alternative
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_2__64_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_alternative_train_64e.yaml
```

##### Passage
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/passage_logical_config_2__64_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_passage_train_64e.yaml
```

#### Config 3 (128 Experts)

##### Alternative
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_3__128_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_alternative_train_128e.yaml
```

##### Passage
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/passage_logical_config_3__128_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_passage_train_128e.yaml
```

#### Config 4 (256 Experts)

##### Alternative
```bash
scalescope-cli \
    --hardware-lib hardware_specs/hardware_library.yaml \
    --physical-topology topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml \
    --logical-topology topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_4__256_experts.yaml \
    --workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
    --model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml \
    --sweep-config sweep_configs/tp_topology_sweep_config.yaml \
    --pass-mode training \
    --output-file hotint25_alternative_train_256e.yaml
```

##### Passage
```bash
scalescope-cli \
--hardware-lib hardware_specs/hardware_library.yaml \
--physical-topology topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml \
--logical-topology topology_zoo/logical/hot_interconnects_2025/passage_logical_config_4__256_experts.yaml \
--workload-spec train_specs/hot_interconnects_2025/train_spec.yaml \
--model-spec model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml \
--sweep-config sweep_configs/tp_topology_sweep_config.yaml \
--pass-mode training \
--output-file hotint25_passage_train_256e.yaml
```

### Reproducing Hot Interconnects'25 results

The unit tests encode the exact configurations used in the paper and assert the expected times for each scenario. To run all of them:

```bash
pytest -q tests/test_hot_interconnects.py
```

Notes:
- The tests construct absolute paths from the repo root and write outputs into a temporary directory.
- See `tests/test_hot_interconnects.py` for the precise combinations of hardware, model, workload, and topology used.
- If you prefer running individual scenarios manually, use the commands in the "Hot Interconnects 2025 configurations" section above, which mirror the test cases.
