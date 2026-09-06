from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).parents[1]
patch = runpy.run_path(str(ROOT / "scripts/prepare_draft_kv_trial.py"))["patch"]


def test_builder_patch_rejects_source_drift():
    with pytest.raises(ValueError, match="Unexpected"):
        patch("pass")


def test_builder_patch_is_scoped_and_compiles():
    source = '''def setup():
    for groups in all_groups:
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
            )
'''
    result = patch(source)
    assert "builder_config = vllm_config" in result
    assert "layer.kv_cache_dtype" in result
    assert "vllm_config=builder_config" in result
    assert "cache_dtype=cache_dtype" in result
    compile(result, "trial", "exec")
    with pytest.raises(ValueError):
        patch(result)


def test_trial_defaults_preserve_profile():
    inner = (ROOT / "scripts/tp4_inner.sh").read_text()
    cluster = (ROOT / "scripts/tp4_cluster.sh").read_text()
    assert 'os.environ.get("DFLASH_KV_DTYPE", "auto")' in inner
    assert 'PROFILE_DECODE:-0' in inner
    assert 'DFLASH_KV_DTYPE:-auto' in cluster
    assert 'if [[ -n "${NCCL_ALGO:-}" ]]' in cluster


def test_standalone_patch_rejects_invalid_block_and_source():
    standalone = runpy.run_path(str(ROOT / "scripts/prepare_standalone_kv_trial.py"))["patch"]
    with pytest.raises(ValueError):
        standalone("pass", 512)
    with pytest.raises(ValueError):
        standalone("pass", 256)


def test_separate_scratch_rows_preserve_default_and_check_capacity():
    source = (ROOT / "overlay/exl3.py").read_text()
    assert 'os.environ.get("EXL3_TEMP_ROWS_DECODE", rows)' in source
    assert "if not 1 <= decode_rows <= rows:" in source
    assert "get_temps(decode_concurrency, decode_rows)" in source
    assert "if tokens <= int(decode_temps[0].shape[1]):" in source
    compile(source, "exl3.py", "exec")
