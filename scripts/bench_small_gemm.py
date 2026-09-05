"""Offline GB10 screening only; no serving monkeypatches."""
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def split_gemm(A, B, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               SPLIT: tl.constexpr, BN: tl.constexpr = 64, BK: tl.constexpr = 64):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    n = tile * BN + tl.arange(0, BN)
    k = split * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.full((16, BN), 0, tl.float32)
    for _ in range(K // SPLIT // BK):
        a = tl.load(A + m[:, None] * K + k[None, :], m[:, None] < M, 0)
        # B is a transposed contiguous [N,K] linear weight.
        b = tl.load(B + n[None, :] * K + k[:, None], n[None, :] < N, 0)
        acc += tl.dot(a, b)
        k += BK
    tl.store(P + split * M * N + m[:, None] * N + n[None, :],
             acc, (m[:, None] < M) & (n[None, :] < N))


def main():
    torch.manual_seed(42)
    for n, k in [(6416, 4096), (4096, 2048), (1024, 4096), (38720, 4096)]:
        a = torch.randn((8, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        reference = a @ w.T
        baseline = triton.testing.do_bench(lambda: a @ w.T)
        previous_backend = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library("cublaslt")
        lt_result = a @ w.T
        lt_error = ((lt_result.float() - reference.float()).norm() / reference.float().norm()).item()
        lt_time = triton.testing.do_bench(lambda: a @ w.T)
        torch.backends.cuda.preferred_blas_library(previous_backend)
        print(json.dumps(dict(n=n, k=k, variant="cublaslt", torch_ms=baseline,
                              trial_ms=lt_time, speedup=baseline / lt_time,
                              relative_l2=lt_error)), flush=True)
        for alignment in [64, 128, 256]:
            padded_n = triton.cdiv(n, alignment) * alignment
            if padded_n == n:
                continue
            padded_w = torch.nn.functional.pad(w, (0, 0, 0, padded_n - n))
            padded_result = (a @ padded_w.T)[:, :n]
            error = ((padded_result.float() - reference.float()).norm()
                     / reference.float().norm()).item()
            elapsed = triton.testing.do_bench(lambda: (a @ padded_w.T)[:, :n])
            print(json.dumps(dict(n=n, k=k, variant="padded_N", padded_n=padded_n,
                                  torch_ms=baseline, trial_ms=elapsed,
                                  speedup=baseline / elapsed, relative_l2=error)), flush=True)
        contiguous_b = w.T.contiguous()
        layout_result = a @ contiguous_b
        truth = a.float() @ w.float().T
        layout_error = ((layout_result.float() - truth).norm() / truth.norm()).item()
        reference_error = ((reference.float() - truth).norm() / truth.norm()).item()
        layout_ms = triton.testing.do_bench(lambda: a @ contiguous_b)
        print(json.dumps(dict(n=n, k=k, variant="contiguous_B",
                              torch_ms=baseline, trial_ms=layout_ms,
                              speedup=baseline / layout_ms,
                              reference_error=reference_error, layout_error=layout_error,
                              numerically_qualified=False)), flush=True)
        for splits in [1, 2, 4, 8, 16]:
            partial = torch.empty((splits, 8, n), device="cuda", dtype=torch.float32)
            bn = int(os.environ.get("TRIAL_BN", "64"))

            def run():
                split_gemm[(triton.cdiv(n, bn), splits)](
                    a, w, partial, 8, n, k, splits, BN=bn, num_warps=4)
                return partial.sum(dim=0).to(torch.bfloat16)

            result = run()
            relative_l2 = ((result.float() - reference.float()).norm()
                           / reference.float().norm()).item()
            assert relative_l2 < 0.01, relative_l2
            elapsed = triton.testing.do_bench(run)
            print(json.dumps(dict(n=n, k=k, splits=splits, bn=bn,
                                  torch_ms=baseline, trial_ms=elapsed,
                                  speedup=baseline / elapsed,
                                  relative_l2=relative_l2)), flush=True)


if __name__ == "__main__":
    main()
