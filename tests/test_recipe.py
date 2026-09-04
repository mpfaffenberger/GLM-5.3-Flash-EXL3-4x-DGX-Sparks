"""Pin recipe.yaml to the profile that passed the full matrix."""
import json
from pathlib import Path

import yaml

RECIPE = yaml.safe_load((Path(__file__).parents[1] / "recipe.yaml").read_text())
CLUSTER = (Path(__file__).parents[1] / "scripts/tp4_cluster.sh").read_text()
DIGEST = "sha256:03161bb433140860c6fbe9505de522f819630215d8eca9a8ba2c73652706dd96"


def test_recipe_pins_validated_image_and_snapshots() -> None:
    assert RECIPE["container"].endswith("@" + DIGEST)
    assert DIGEST in CLUSTER, "recipe and tp4_cluster.sh must pin the same image"
    d = RECIPE["defaults"]
    assert d["revision"] == "25a44fdbf16862a46b7cc9921142c6c81350af2f"
    spec = json.loads(d["speculative_config"])
    assert spec["model"] == "incoai/GLM-5.3-Flash-DFlash2"
    assert spec["revision"] == "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
    assert spec["num_speculative_tokens"] == 7
    assert spec["draft_tensor_parallel_size"] == 4


def test_recipe_matches_validated_profile() -> None:
    d = RECIPE["defaults"]
    assert d["gpu_memory_utilization"] == 0.55
    assert d["tensor_parallel"] == 4
    assert d["max_num_seqs"] == 10
    assert d["max_num_batched_tokens"] == 2048
    assert d["kv_cache_dtype"] == "fp8"
    assert json.loads(d["compilation_config"]) == {"cudagraph_mode": "NONE"}
    env = RECIPE["env"]
    assert env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
    assert env["EXL3_MOE_CONCURRENCY"] == "6"
    assert env["EXL3_MOE_DECODE_CONCURRENCY"] == "6"
    assert RECIPE["runtime"] == "vllm-distributed"
    assert RECIPE["cluster_only"] is True
    assert RECIPE["min_nodes"] == 4


def test_recipe_command_survives_sparkrun_plumbing() -> None:
    cmd = RECIPE["command"]
    # sparkrun appends its own distributed flags; the template must not.
    for flag in ("--nnodes", "--node-rank", "--headless", "--master-port"):
        assert flag not in cmd
    # sparkrun's placeholder regex is a non-greedy {...}; the wrapper must
    # not use braces so the JSON defaults are the only substitutions.
    wrapper = cmd.split("serve() (")[0]
    assert "{" not in wrapper.replace("{roce_if}", "").replace("{roce_hca}", "").replace("{master_addr}", "")
    assert 'exec vllm serve "$@" --master-addr "$MASTER_ADDR"' in cmd
    assert cmd.rstrip().endswith("'{speculative_config}'")
    assert '.strip() == "RoCE v2"' in cmd, "must select the RoCE v2 GID, not the v1 twin"
    ec = RECIPE["executor_config"]
    assert ec["entrypoint"] == "", "base image ENTRYPOINT swallows sparkrun's bash -c"
    assert ec["user"] == "root"
    assert "IPC_LOCK" in ec["cap_add"]
