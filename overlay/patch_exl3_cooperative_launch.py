#!/usr/bin/env python3
"""Make ExLlamaV3's software-barrier MoE launch cooperative and self-contained.

The kernel synchronizes CUDA blocks through device-global atomics, so every
block must be resident together; cooperative launch guarantees that. The
barrier/tile-lock arena is also moved off the DevCtx arena shared with the
GEMM kernels onto a once-allocated MoE-private arena per device and stream,
cleared with stream-ordered memsets before each launch. After each stream's
first use, no driver allocations happen on the launch path.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3")
    target = root / "exllamav3/exllamav3_ext/quant/exl3_moe.cu"
    text = target.read_text()
    locks_old = "    int* locks = DevCtx::instance().get_locks(device);"
    locks_new = """    // The DevCtx lock arena is shared with the GEMM kernels. Give the MoE
    // kernel its own per-stream arena, allocated once: per-launch
    // cudaMallocAsync/cudaFreeAsync churns driver allocations on the launch
    // path and stalls under GB10 unified-memory pressure. Launches on each
    // stream serialize, so stream-ordered clears of the slots this launch
    // touches (tile locks per group plus barrier counter/sense pairs) are
    // sufficient.
    int* locks = moe_private_locks(device, stream);
    size_t tile_lock_bytes =
        concurrency * (MAX(hidden_dim, intermediate_dim) / 128) * sizeof(int);
    cuda_check(cudaMemsetAsync(locks, 0, tile_lock_bytes, stream));
    cuda_check(cudaMemsetAsync(
        locks + BARRIER_LOCKS_OFFSET, 0, concurrency * 2 * sizeof(int), stream));"""
    helper_anchor = "std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};\n"
    helper = """std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

// MoE-private lock arena per device and stream, allocated on first use. vLLM
// may execute model work on more than one stream; sharing one arena across
// those streams recreates the counter/sense race this isolation fixes.
static int* moe_private_locks(int device, cudaStream_t stream)
{
    static std::map<cudaStream_t, int*> arenas[MAX_DEVICES];
    static std::mutex arena_mutexes[MAX_DEVICES];
    std::lock_guard<std::mutex> guard(arena_mutexes[device]);
    auto it = arenas[device].find(stream);
    if (it == arenas[device].end())
    {
        constexpr size_t locks_bytes =
            (MAX_TILES_C + MAX_BARRIERS * 2) * sizeof(int);
        int* arena;
        cuda_check(cudaMalloc((void**) &arena, locks_bytes));
        it = arenas[device].emplace(stream, arena).first;
    }
    return it->second;
}
"""
    if text.count(helper_anchor) != 1:
        raise SystemExit(f"{target}: expected one kernel attribute set definition")
    text = text.replace("#include <set>\n", "#include <set>\n#include <map>\n#include <mutex>\n", 1)
    text = text.replace(helper_anchor, helper, 1)
    if text.count(locks_old) != 1:
        raise SystemExit(f"{target}: expected one shared lock arena target")
    text = text.replace(locks_old, locks_new, 1)
    old = """    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
"""
    new = """    // This kernel uses software inter-block barriers. A normal launch may
    // schedule only part of the grid when another stream occupies an SM,
    // deadlocking resident blocks. Cooperative launch guarantees co-residency.
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
"""
    if text.count(old) != 1:
        raise SystemExit(f"{target}: expected one cudaLaunchKernel target")
    text = text.replace(old, new, 1)
    target.write_text(text)
    print(f"patched cooperative EXL3 MoE launch with private locks: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
