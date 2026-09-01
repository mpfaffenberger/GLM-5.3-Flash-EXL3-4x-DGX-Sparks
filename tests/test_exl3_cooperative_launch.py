from pathlib import Path
import subprocess
import sys


PATCHER = Path(__file__).parents[1] / "overlay/patch_exl3_cooperative_launch.py"


def test_replaces_exactly_one_launch_and_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "exllamav3/exllamav3_ext/quant/exl3_moe.cu"
    target.parent.mkdir(parents=True)
    target.write_text(
        """void launch() {
    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
}
"""
    )

    first = subprocess.run(
        [sys.executable, str(PATCHER), str(tmp_path)], capture_output=True, text=True
    )
    assert first.returncode == 0, first.stderr
    patched = target.read_text()
    assert "cudaLaunchCooperativeKernel" in patched
    assert "cudaLaunchKernel\n" not in patched

    second = subprocess.run(
        [sys.executable, str(PATCHER), str(tmp_path)], capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "expected one cudaLaunchKernel target" in second.stderr
