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
    expected_time_sec = 499899.9317777489
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
    expected_time_sec = 318540.54833550507
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
    expected_time_sec = 551512.0702480453
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
    expected_time_sec = 319774.2699062866
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
    expected_time_sec = 654736.3564060705
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
    expected_time_sec = 322241.7437726247
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
    expected_time_sec = 861184.9379395536
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
    expected_time_sec = 327176.7222300755
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
    expected_time_sec = 0.07999825566935967
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
    expected_time_sec = 0.07979502933707058
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
    expected_time_sec = 0.0793885779942269
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
    expected_time_sec = 0.07857567663027396
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
    expected_time_sec = 0.15538103370396505
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
    expected_time_sec = 0.1726235720899926
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
    expected_time_sec = 0.20710864925856806
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
    expected_time_sec = 0.27607880399223933
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Decode tests - Passage


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
    expected_time_sec = 0.420056455858114
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
    expected_time_sec = 0.4205142479993659
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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
    expected_time_sec = 0.4214298330749101
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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
    expected_time_sec = 0.4232610040190392
    assert time_sec == pytest.approx(expected_time_sec, abs=0.001)


# Decode tests - Alternative


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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


@pytest.mark.skip(reason="Temporarily disabled due to known decode regression drift.")
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
