#!/usr/bin/env python3
"""Backport FlashInfer's alternating sparse-MLA pipeline barriers.

The math warps use non-blocking named-barrier arrivals and can advance one
iteration ahead of the IO warps. Reusing one barrier ID lets a late arrival
desynchronize its next phase, leaving a single CTA's IO warps in BAR.SYNC.
FlashInfer upstream issue #3700 fixes this by alternating IDs 1 and 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

INCLUDE_ROOT = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/"
    "attention/sparse_mla_sm120"
)

HELPER_ANCHOR = """// mbarrier (SM90+) for async copy tracking
"""
HELPERS = """// Producer/consumer handshakes where the arriving side does not block must
// alternate between two barrier ids: the non-blocking side may be one iteration
// ahead, and arriving twice in one phase desynchronizes the barrier (#3700).
template <int ID_EVEN, int ID_ODD, int CNT>
__device__ __forceinline__ void bar_arrive_alt(int parity) {
  if (parity)
    bar_arrive_t<ID_ODD, CNT>();
  else
    bar_arrive_t<ID_EVEN, CNT>();
}

template <int ID_EVEN, int ID_ODD, int CNT>
__device__ __forceinline__ void bar_sync_alt(int parity) {
  if (parity)
    bar_sync_t<ID_ODD, CNT>();
  else
    bar_sync_t<ID_EVEN, CNT>();
}

"""


def patch(root: Path = INCLUDE_ROOT) -> bool:
    barrier = root / "arch/barrier.cuh"
    kernel = root / "prefill_kernel.cuh"
    barrier_text = barrier.read_text()
    kernel_text = kernel.read_text()

    if HELPERS in barrier_text:
        if kernel_text.count("bar_sync_alt<1, 5, BLOCK_THREADS>(ti & 1);") != 3:
            raise SystemExit(f"{kernel}: partial alternating-barrier patch")
        if kernel_text.count("bar_arrive_alt<1, 5, BLOCK_THREADS>(ti & 1);") != 2:
            raise SystemExit(f"{kernel}: partial alternating-barrier patch")
        print(f"already patched: {root}")
        return False

    if barrier_text.count(HELPER_ANCHOR) != 1:
        raise SystemExit(f"{barrier}: expected one helper anchor")
    if kernel_text.count("bar_sync_t<1, BLOCK_THREADS>();") != 3:
        raise SystemExit(f"{kernel}: expected three blocking barrier calls")
    if kernel_text.count("bar_arrive_t<1, BLOCK_THREADS>();") != 2:
        raise SystemExit(f"{kernel}: expected two arrival barrier calls")

    barrier.write_text(
        barrier_text.replace(HELPER_ANCHOR, HELPERS + HELPER_ANCHOR, 1)
    )
    kernel.write_text(
        kernel_text.replace(
            "bar_sync_t<1, BLOCK_THREADS>();",
            "bar_sync_alt<1, 5, BLOCK_THREADS>(ti & 1);",
        ).replace(
            "bar_arrive_t<1, BLOCK_THREADS>();",
            "bar_arrive_alt<1, 5, BLOCK_THREADS>(ti & 1);",
        )
    )
    print(f"patched alternating sparse-MLA barriers: {root}")
    return True


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else INCLUDE_ROOT
    patch(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
