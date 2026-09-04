# EXL3 fused-MoE wedge investigation

## Reproducer

The four-node TP=4 profile completed the benchmark matrix until the final
65,535-token, concurrency-10 batch. One request emitted a token; the other
nine never emitted one. All GPUs then remained near 96% utilization at low
power and all workers stopped making generation progress.

Do not use `/v1/models` as a progress probe. The API process remains healthy
while its compute workers are stuck.

## Isolation result

The same 65K x C10 shape completed 20/20 requests in eager, non-speculative
mode when only decode-sized MoE batches used the non-fused EXL3 expert path.
Prefill continued to use the fused path. Artifact:

```text
results/isolate-decode-loop-65k-c10-20260901-002301/
```

This isolates the EXL3 failure to its fused decode launch rather than prompt
construction, KV capacity, or the scheduler alone.

## Kernel hazard

ExLlamaV3's `exl3_moe` kernel uses eight CUDA blocks per expert group and a
software global barrier (`group_barrier`) backed by device-global atomics.
On GB10 the extension reports six concurrent groups, producing a 48-block
grid on a 48-SM GPU.

That launch assumes every barrier participant can become resident together.
It has no occupancy slack. If attention, NCCL, or another stream occupies even
one SM, resident blocks can wait forever for a barrier participant that cannot
be scheduled. The observed high-utilization/low-power spin is consistent with
this failure mode.

## Candidate fix

`overlay/exl3.py` now supports separate fused-MoE scratch pools:

- `EXL3_MOE_CONCURRENCY`: prefill expert groups (default 6)
- `EXL3_MOE_DECODE_CONCURRENCY`: decode expert groups (default 1)

Decode therefore launches one eight-block group and processes active experts
serially, while prefill retains the faster six-group path. This avoids the
unsafe full-SM software barrier without falling back to the much slower Python
expert loop.

The split-pool candidate completed a 65K x C10 eager/no-spec gate with 20/20
requests and no errors. The shortened TG=16 gate artifact is:

```text
results/exl3-split-concurrency-tg16-65k-c10-20260901-011819/
```

`EXL3_FUSED_MOE_DECODE=0` remains an isolation switch, not the intended
production setting.

## Cooperative-launch fix

The stronger fix changes ExLlamaV3's launch from `cudaLaunchKernel` to
`cudaLaunchCooperativeKernel`. This makes CUDA enforce the kernel's existing
co-residency requirement instead of approximating it from the SM count.

`overlay/patch_exl3_cooperative_launch.py` applies the source patch before the
extension is compiled. The resulting binary imports
`cudaLaunchCooperativeKernel@libcudart.so.13` and starts successfully on GB10
with all six decode groups enabled.

A clean eager/no-spec 65K x C10 gate completed 20/20 requests with six fused
decode groups and no errors:

```text
results/exl3-coop-clean-tg16-65k-c10-20260901-023751/
```

The subsequent full production matrix still wedged after C1/C2 at 65K, on the
first C5 batch. Live stacks on all surviving workers again stopped in
`apply_exl3_fused_moe`. Clearing the extension's global lock arena before every
launch did not help: a fresh production-shaped C5 run wedged immediately, so
stale accumulated lock state was falsified.

The decisive isolation was execution mode:

- eager + DFlash2 completed the 65K x C5 gate (10/10, no errors);
- CUDA graphs without speculation stalled during startup profiling;
- the graph-startup ranks diverged across KDA, MLA, and sparse-MLA paths.

The supported candidate therefore disables CUDA graphs while retaining
torch.compile (`--compilation-config '{"cudagraph_mode":"NONE"}'`). Full-matrix
validation is required before promoting that profile.

Reducing concurrent groups to five or four and reducing blocks per group from
eight to four still failed. The underlying barrier arena came from
`DevCtx::get_locks(device)` and was shared by every layer and CUDA stream. Two
overlapping MoE launches therefore used the same counter/sense slots. An early
extension patch allocated, cleared, and released a private lock arena for each
launch with stream-ordered `cudaMallocAsync` / `cudaFreeAsync`.

That private-arena hypothesis also failed full accumulated validation. A
six-group compiled/no-graph run passed a chained 65K C5+C10 gate (30/30), then
the full matrix advanced beyond the prior five-group failure boundary before
wedging after 426 clean tracked completions. All four rank stacks were again
inside `apply_exl3_fused_moe`. Private arenas fix cross-launch aliasing but do
not make the fused kernel safe for production.

The per-launch allocation strategy also churned the CUDA driver's allocator on
every MoE layer. On GB10 that competes with unified host/device memory. During
a later 65K x C10 run at `GPU_MEM_UTIL=0.75`, the head reached 112/121 GiB used,
started swapping, and logged `NV_ERR_NO_MEMORY` from
`_memdescAllocInternal`. CUDA-GDB then showed one rank's resident NCCL ring
waiting at `BAR.SYNC`, while the other ranks had no resident kernels and were
blocked inside `cuLaunchKernel*` driver locks. Those stacks were a consequence
of allocation failure, not proof that NCCL or sparse MLA caused the wedge.

`GPU_MEM_UTIL=0.60` leaves roughly 27 GiB available and avoids those driver OOM
events, but it still wedged after 56/60 requests in a three-pass 65K x C10
gate. Memory exhaustion was therefore a real independent failure mode, not the
complete explanation.

The next candidate kept one MoE-private arena per device for the process
lifetime. Before each launch it cleared only the tile-lock and barrier regions
with stream-ordered memsets. This preserved isolation from the shared DevCtx
arena without repeated allocation on the launch path.

The first direct A/B gate passed. With the old per-launch allocator and
`GPU_MEM_UTIL=0.60`, the third 65K x C10 pass wedged after 56/60 requests. With
the static arena and otherwise identical settings, all 60/60 requests completed
with no HTTP or stream errors:

```text
results/exl3-static-arena-mem060-65k-c10-20260903-161037/
```

The same image then completed a 100K x C10 context-load/inference gate with
20/20 requests and no errors:

```text
results/exl3-static-arena-mem060-100k-c10-20260903-172722/
```

This promotes the static arena from a code-only candidate to a successful
targeted fix. It did not pass the full matrix: after 82 successful requests,
rank 0 failed at 4K x C5 while launching DeepGEMM's MHC pre-normalization GEMM
with `CUDA_ERROR_NOT_PERMITTED`. The engine exited and the remaining 620
requests were connection failures:

```text
results/exl3-static-arena-mem060-full-20260903-182611/
```

That per-device arena was still shared by every CUDA stream, recreating the
same counter/sense race whenever another model-execution stream overlapped a
MoE launch. The current candidate owns one arena per `(device, cudaStream_t)`.
Each stream serializes its own clear and cooperative launch while independent
streams cannot alias barrier state. The first use of each stream allocates one
arena; subsequent launches perform no driver allocation. It still requires
the full matrix below.

The per-stream candidate completed two chained accumulated gates on one engine:

```text
results/exl3-stream-arena-accum-0-4k-c125-20260903-185659/       # 72/72
results/exl3-stream-arena-accum-0-4k-c12510-20260903-190852/     # 162/162
```

That is 234 consecutive requests with no errors and crosses both the 82-request
full-matrix crash of the per-device arena and its 100K targeted gates. Full
matrix validation remains the promotion requirement.

The following full matrix crossed the previous 426-completion boundary, then
wedged at 32K x C10 after 481 completions:

```text
results/exl3-stream-arena-mem060-full-20260903-192519/
```

Async CUDA-GDB finally isolated this remaining failure. Ranks 0, 1, and 3 were
waiting in NCCL AllReduce while rank 2 had one block resident in FlashInfer's
`sparse_mla_prefill_kernel<ModelType=2, ComputeMode=0, 16, 2048, 64>`. Only
the block's four IO warps remained active, all at one PC. KV occupancy was
roughly 8%, so this failure was neither KV exhaustion nor the EXL3 MoE arena.
The capture is in:

```text
/home/mpfaffenberger/glm53-hang-dumps/exl3-stream-arena-async-32k-c10-20260903-205957/
```

DFlash-7 can expand a C10 target batch beyond FlashInfer's 64-token standalone
decode cutoff, routing it through the problematic prefill orchestrator. The
next gate reduces DFlash draft depth so C10 remains on the decode path.

That hypothesis was incomplete. DFlash-3 passed 32K x C10 at 60/60, but the
subsequent 65K x C10 gate wedged after 13 completions. CUDA-GDB reproduced the
same split: ranks 0, 1, and 3 in NCCL and rank 2 in a single surviving
`sparse_mla_prefill_kernel` block. Long prompt chunks use the prefill kernel
regardless of speculative decode depth.

FlashInfer upstream commit `53779166d09ada303ecdd8043277d1a8b88020fb`
contains the matching fix (issue #3700): the non-blocking math-warps side can
advance one iteration ahead, so reusing named barrier 1 lets arrivals from two
phases desynchronize it. Upstream alternates named barriers 1 and 5 according
to tile parity. `overlay/patch_flashinfer_sparse_mla_barrier.py` backports only
the two helper functions and five prefill-pipeline call-site changes into the
pinned FlashInfer headers. The generated `sparse_mla_sm120` JIT cache must be
removed once after applying the patch so FlashInfer recompiles the cubin.

The first barrier-patched image still wedged at 65K x C10 after 52/60, but
inspection showed that it had loaded the base image's bundled precompiled
`flashinfer_jit_cache/jit_cache/sparse_mla_sm120.so`; patched headers alone do
not invalidate that module. The image now removes that bundled module after
patching. All four ranks then generated new 923800-byte modules under
`/root/.cache/flashinfer/0.6.17/121a/cached_ops/sparse_mla_sm120/` during the
same startup.

The first gate against those definitely rebuilt cubins passed:

```text
results/exl3-barrier-v2-mem060-100k-c10-20260904-000532/  # 20/20, zero errors
```

This is positive targeted evidence; full-matrix validation is still required.

The same warm engine then failed a full matrix after 32 completions in the
depth-0 C10 cell. Rank 3 reported `CUBLAS_STATUS_INTERNAL_ERROR` from the BF16
KDA output projection and lost its CUDA context; ranks 0-2 remained in NCCL.
No Xid or NVRM allocation error was logged. Because this followed the 100K
soak and surfaced as a rank-local driver/library failure, the next full-matrix
trial starts from a clean reset at `GPU_MEM_UTIL=0.55` for additional UMA
headroom. The 0.60 candidate is not promoted.

## Final result

The clean-reset full matrix at `GPU_MEM_UTIL=0.55` passed:

```text
results/exl3-barrier-v2-mem055-full-20260904-012121/   # 702/702, zero errors
```

That run used DFlash-7, six MoE groups, `cudagraph_mode=none`, and covered
every depth from 0 through 100K at concurrencies 1 through 10, crossing every
prior failure boundary (82, 426, 481, 52-of-60 at 65K x C10, and 32 on the
post-soak warm engine).

Three independent defects were fixed; none alone was sufficient:

1. **GB10 unified-memory exhaustion.** `GPU_MEM_UTIL` above about 0.60 lets
   the NVIDIA driver fail allocations on the launch path with no Python-visible
   OOM. Symptom: one rank in NCCL `BAR.SYNC`, peers blocked in `cuLaunchKernel`
   driver locks, or a rank-local `CUBLAS_STATUS_INTERNAL_ERROR`. Fix: 0.55.
2. **EXL3 fused-MoE lock arena aliasing.** Per-launch `cudaMallocAsync` of the
   lock arena churned driver allocations under that same memory pressure, and a
   per-device static arena was still shared across CUDA streams. Fix: one
   arena per `(device, cudaStream_t)`, cleared with stream-ordered memsets
   (`overlay/patch_exl3_cooperative_launch.py`), plus
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.
3. **FlashInfer SM120 sparse-MLA prefill barrier race.** Upstream #3700: the
   non-blocking math-warp arrivals could double-arrive on named barrier 1 and
   starve the IO warps. Fix: alternate barriers 1 and 5 by tile parity
   (`overlay/patch_flashinfer_sparse_mla_barrier.py`), and delete the base
   image's prebuilt `flashinfer_jit_cache/.../sparse_mla_sm120` so the JIT
   actually recompiles from the patched header.

## Promotion gate

The candidate was not promoted until it completed:

1. 65K x C10 targeted gate (60/60);
2. 100K x C10 targeted gate (20/20);
3. the complete benchmark matrix through 100K x C10 with progress-based stall
   detection (702/702).

Isolated cell passes were never accepted as sufficient; every intermediate
candidate that passed a targeted gate subsequently failed the full matrix.
