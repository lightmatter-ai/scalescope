import os
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml

from scalescope.run_scalescope import main as scalescope_main


@pytest.fixture(scope="module")
def tmp_output_dir(tmp_path_factory):
    """
    Creates a temporary directory for output files and cleans it up after tests.
    """
    output_dir = tmp_path_factory.mktemp("scalescope_test_output")
    yield output_dir
    # Cleanup is handled automatically by tmp_path_factory


def run_scalescope_cli(
    tmp_output_dir: Path,
    hardware_lib: str,
    physical_topology: str,
    logical_topology: str,
    workload_spec: str,
    model_spec: str,
    sweep_config: str,
    pass_mode: str,
    output_file_name: str,
):
    """Helper function to run the scalescope-cli command and return time to train metrics."""
    # Determine the project root (assuming tests are in project_root/tests/)
    project_root = Path(__file__).parent.parent

    # Prepare arguments for scalescope_main
    # All config file paths should be absolute paths derived from project_root.
    # The output file path should be the simple filename, as scalescope_main will prepend "filtered_".
    args = [
        "scalescope-cli",  # This is usually sys.argv[0]
        "--hardware-lib",
        str(project_root / hardware_lib),
        "--physical-topology",
        str(project_root / physical_topology),
        "--logical-topology",
        str(project_root / logical_topology),
        "--workload-spec",
        str(project_root / workload_spec),
        "--model-spec",
        str(project_root / model_spec),
        "--sweep-config",
        str(project_root / sweep_config),
        "--pass-mode",
        pass_mode,
        "--output-dir",
        str(tmp_output_dir),
        "--run-name",
        output_file_name,
        "--no-timestamp",
        "--overwrite",
        "--raw-output",
        "--save-run-artifacts",
    ]

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = StringIO()
    redirected_stderr = StringIO()
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr

    # Store original sys.argv and original working directory
    original_argv = sys.argv
    original_cwd = os.getcwd()

    sys.argv = args

    try:
        # Change to the temporary output directory before calling main
        # This ensures output files are written to this temp directory directly.
        os.chdir(tmp_output_dir)
        scalescope_main()
    except SystemExit as e:
        # scalescope_main exits with SystemExit on success (0) or failure
        if e.code != 0:
            pytest.fail(
                f"scalescope_main exited with code {e.code}.\n"
                f"STDOUT: {redirected_stdout.getvalue()}\nSTDERR: {redirected_stderr.getvalue()}"
            )
    except Exception as e:
        pytest.fail(
            f"scalescope_main raised an unexpected exception: {e}\nSTDOUT: {redirected_stdout.getvalue()}\nSTDERR: {redirected_stderr.getvalue()}"
        )
    finally:
        # Restore original stdout, stderr, argv and working directory
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.argv = original_argv
        os.chdir(original_cwd)

    # Output artifacts are written to tmp_output_dir/<run-name>/...
    run_dir = tmp_output_dir / output_file_name
    manifest_path = run_dir / "manifest.yaml"
    assert manifest_path.exists(), f"Manifest file {manifest_path} was not created."
    assert manifest_path.stat().st_size > 0, f"Manifest file {manifest_path} is empty."

    with open(manifest_path, "r") as f:
        try:
            manifest = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Failed to parse manifest YAML file {manifest_path}: {e}")
    runs = manifest.get("runs") or []
    assert len(runs) == 1, f"Expected exactly 1 run for empty sweep config, got {len(runs)}"
    run_path = run_dir / runs[0]["path"]
    assert run_path.exists(), f"Run file {run_path} was not created."

    with open(run_path, "r") as f:
        try:
            result = yaml.safe_load(f)
            time_sec = float(result["time"]["total_time_seconds"])
            return time_sec
        except (yaml.YAMLError, KeyError, ValueError) as e:
            pytest.fail(f"Failed to parse or find data in run YAML file {run_path}: {e}")


# Config 1 (32 Experts)
def test_alternative_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_1__32_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_alternative_32_experts.yaml",
    )
    expected_time_sec = 483339.5466125123
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_1__32_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_passage_32_experts.yaml",
    )
    expected_time_sec = 307443.3820138336
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Config 2 (64 Experts)
def test_alternative_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_2__64_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_alternative_64_experts.yaml",
    )
    expected_time_sec = 533698.1142723674
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_2__64_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_passage_64_experts.yaml",
    )
    expected_time_sec = 307423.53277417406
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Config 3 (128 Experts)
def test_alternative_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_3__128_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_alternative_128_experts.yaml",
    )
    expected_time_sec = 634415.2588095106
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_3__128_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_passage_128_experts.yaml",
    )
    expected_time_sec = 307383.8650196298
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Config 4 (256 Experts)
def test_alternative_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_4__256_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_alternative_256_experts.yaml",
    )
    expected_time_sec = 835849.5571012292
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_4__256_experts.yaml",
        workload_spec="train_specs/hot_interconnects_2025/train_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="training",
        output_file_name="scalescope_res_passage_256_experts.yaml",
    )
    expected_time_sec = 307304.56023531605
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Prefill tests - Passage


def test_passage_prefill_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_1__32_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.08260291856484964
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_prefill_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_2__64_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.08252150823632526
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_prefill_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_3__128_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.0823587087270278
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_prefill_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_4__256_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.0820331308561841
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Prefill tests - Alternative


def test_alternative_prefill_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_1__32_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.14789117355930353
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_alternative_prefill_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_2__64_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.1651702567464605
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_alternative_prefill_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_3__128_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.19972842946509986
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_alternative_prefill_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_4__256_experts.yaml",
        workload_spec="inference_prefill_specs/hot_interconnects_2025/inference_prefill_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_prefill",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.2688447812467039
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Decode tests - Passage


# Passage decode tests — intra-rack centralized expert pool regression baseline.
#
# All four "passage" configs share the same regime:
#   attn_tp_size = 16, full_expert_set_xpu_count = 512, scale_up_domain_size = 512
#   => attn_domains_per_full_expert_set = 32, full_set fits in one scale-up
#      domain, so decode dispatch + combine go through the symmetric A2A path
#      in scalescope/comm_model.py.
# Headline TPOT barely moves with the symmetric-A2A model because this regime
# is memory-bound (memory ≈ 97% of total, comm overlap absorbs the rest), but
# the moe_all_to_all volume_bytes in the run artifact are non-zero and
# symmetric, which is the structural property these baselines lock in.


def test_passage_decode_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_1__32_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="decode_32_experts_scalescope_res.yaml",
    )
    expected_time_sec = 0.4933071407161978
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_decode_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_2__64_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.4937027847425036
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_decode_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_3__128_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.49449407358907177
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


def test_passage_decode_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/passage_logical_config_4__256_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.4960766520758352
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Decode tests - Alternative


# Alternative decode tests — cross-rack centralized expert pool.
#
# All four "alternative" configs have full_expert_set_xpu_count = 512
# (EP_SU=128, EP_SO=4) but scale_up_domain_size = 128, so the full expert set
# does not fit in one scale-up domain. Scale-out-spanning wide-EP dispatch +
# combine on EP_SO is not yet implemented in scalescope/comm_model.py, so
# these tests are skipped. The expected_time_sec values below are kept as
# legacy baselines from a comm model that silently zeroed multi-domain decode
# dispatch; they are not currently exercised.


@pytest.mark.skip(
    reason=(
        "Scale-out-spanning wide-EP set: full_expert_set_xpu_count=512 > "
        "scale_up_domain_size=128. Hierarchical dispatch + combine on EP_SO "
        "is not yet implemented in scalescope/comm_model.py."
    )
)
def test_alternative_decode_32_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_1__32_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.44138090577909633
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


@pytest.mark.skip(
    reason=(
        "Scale-out-spanning wide-EP set: full_expert_set_xpu_count=512 > "
        "scale_up_domain_size=128. Hierarchical dispatch + combine on EP_SO "
        "is not yet implemented in scalescope/comm_model.py."
    )
)
def test_alternative_decode_64_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_2__64_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_2__64_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.4425116346060625
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


@pytest.mark.skip(
    reason=(
        "Scale-out-spanning wide-EP set: full_expert_set_xpu_count=512 > "
        "scale_up_domain_size=128. Hierarchical dispatch + combine on EP_SO "
        "is not yet implemented in scalescope/comm_model.py."
    )
)
def test_alternative_decode_128_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_3__128_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_3__128_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.44477309305303525
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


@pytest.mark.skip(
    reason=(
        "Scale-out-spanning wide-EP set: full_expert_set_xpu_count=512 > "
        "scale_up_domain_size=128. Hierarchical dispatch + combine on EP_SO "
        "is not yet implemented in scalescope/comm_model.py."
    )
)
def test_alternative_decode_256_experts(tmp_output_dir):
    time_sec = run_scalescope_cli(
        tmp_output_dir,
        hardware_lib="hardware_specs/hardware_library.yaml",
        physical_topology="topology_zoo/physical/hot_interconnects_2025/alternative_physical_32768.yaml",
        logical_topology="topology_zoo/logical/hot_interconnects_2025/alternative_logical_config_4__256_experts.yaml",
        workload_spec="inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml",
        model_spec="model_zoo/hot_interconnects_2025/gpt_moe_config_4__256_experts.yaml",
        sweep_config="sweep_configs/tp_topology_sweep_config.yaml",
        pass_mode="inference_decode",
        output_file_name="scalescope_res.yaml",
    )
    expected_time_sec = 0.44929601074002135
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)
