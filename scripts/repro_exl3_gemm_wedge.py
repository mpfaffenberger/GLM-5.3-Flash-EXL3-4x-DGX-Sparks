#!/usr/bin/env python3
"""Single-GPU stress reproducer for the ExLlamaV3 GEMM software-barrier wedge.

Hammers LinearEXL3.forward (exl3_gemm / BC graph path) with GLM-5.3-Flash
shapes and randomized row counts. Modes:

  single  - one CUDA stream (tests the barrier itself)
  dual    - two CUDA streams issuing concurrently (tests arena sharing)

A wedge shows up as the process never printing DONE; the caller enforces a
timeout. Exit code 0 = completed, anything else = wedge or error.

Usage: repro_exl3_gemm_wedge.py MODE [iters] [seed]
"""
from __future__ import annotations

import os
import sys
import time

import torch

from vllm.model_executor.layers.quantization import exl3 as overlay  # noqa: E402

K_BITS = 4
# (in_features, out_features) drawn from GLM-5.3-Flash TP4 shards.
SHAPES = [
    (4096, 1536),   # q_a_proj
    (4096, 512),    # kv_a_proj
    (4096, 4096),   # o_proj-ish
    (4096, 2048),   # routed expert gate/up (moe_intermediate)
    (2048, 4096),   # routed expert down
    (4096, 3072),   # shared expert slice (12288 / 4)
    (3072, 4096),
]
ROW_CHOICES = [1, 2, 4, 8, 16, 64, 128, 144, 145, 256, 512, 1024, 2048, 4096]


def make_layer(inf: int, outf: int, gen: torch.Generator):
    trellis = torch.randint(
        -32768, 32767, (inf // 16, outf // 16, K_BITS * 16),
        dtype=torch.int16, device="cuda", generator=gen,
    )
    suh = torch.where(torch.rand(inf, device="cuda", generator=gen) < 0.5, -1.0, 1.0).half()
    svh = torch.where(torch.rand(outf, device="cuda", generator=gen) < 0.5, -1.0, 1.0).half()
    mcg = torch.tensor([0x1EB5], dtype=torch.int32, device="cuda")
    return overlay.make_linear_exl3(trellis, suh, svh, mcg)


def worker(layers, iters: int, seed: int, tag: str) -> None:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for i in range(iters):
        layer = layers[int(torch.randint(len(layers), (1,), generator=rng))]
        rows = ROW_CHOICES[int(torch.randint(len(ROW_CHOICES), (1,), generator=rng))]
        x = torch.randn(rows, layer.in_features, device="cuda", dtype=torch.half)
        y = layer.forward(x, {}, out_dtype=torch.float32)
        if i % 500 == 0:
            torch.cuda.synchronize()
            print(f"[{tag}] iter {i} rows={rows} ok {time.time() - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()


def main() -> int:
    mode = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layers = [make_layer(i, o, gen) for i, o in SHAPES]
    torch.cuda.synchronize()
    print(f"mode={mode} iters={iters} seed={seed} layers={len(layers)}", flush=True)

    if mode == "single":
        worker(layers, iters, seed, "s0")
    elif mode in ("dual", "dual-iso"):
        import threading

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        # dual-iso: private layer objects per thread, so only the device-side
        # lock arena is shared. Plain dual also shares host state.
        per_thread = [layers, layers]
        if mode == "dual-iso":
            per_thread = [layers, [make_layer(i, o, gen) for i, o in SHAPES]]
            torch.cuda.synchronize()

        def run(idx: int) -> None:
            with torch.cuda.stream(streams[idx]):
                worker(per_thread[idx], iters, seed + idx, f"s{idx}")

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        raise SystemExit(f"unknown mode {mode}")

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("EXL3_FUSED_MOE", "0")
    sys.exit(main())
