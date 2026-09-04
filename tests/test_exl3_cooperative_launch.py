from pathlib import Path
import subprocess
import sys


PATCHER = Path(__file__).parents[1] / "overlay/patch_exl3_cooperative_launch.py"


def test_replaces_exactly_one_launch_and_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "exllamav3/exllamav3_ext/quant/exl3_moe.cu"
    target.parent.mkdir(parents=True)
    target.write_text(
        """#include <set>
std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};
void launch() {
    int* locks = DevCtx::instance().get_locks(device);
    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    cuda_check(cudaPeekAtLastError());
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
    assert "moe_private_locks(device, stream)" in patched
    assert "cudaMemsetAsync" in patched
    assert "std::map<cudaStream_t, int*> arenas[MAX_DEVICES]" in patched
    assert "cudaMalloc((void**) &arena" in patched
    assert "#include <mutex>" in patched
    assert "cudaMallocAsync(" not in patched
    assert "cudaFreeAsync(" not in patched
    assert "get_locks(device)" not in patched
    # The helper must be defined before the launcher that calls it.
    assert patched.index("static int* moe_private_locks") < patched.index(
        "moe_private_locks(device, stream);"
    )

    second = subprocess.run(
        [sys.executable, str(PATCHER), str(tmp_path)], capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "expected one shared lock arena target" in second.stderr
