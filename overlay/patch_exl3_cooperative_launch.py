#!/usr/bin/env python3
"""Make ExLlamaV3's software-barrier MoE launch cooperative.

The kernel synchronizes CUDA blocks through device-global atomics, so every
block must be resident together. A normal launch cannot guarantee that when
another stream owns an SM. Cooperative launch provides that guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3")
    target = root / "exllamav3/exllamav3_ext/quant/exl3_moe.cu"
    text = target.read_text()
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
    target.write_text(text.replace(old, new, 1))
    print(f"patched cooperative EXL3 MoE launch: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
