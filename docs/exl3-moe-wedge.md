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

## Promotion gate

Do not promote the candidate merely because a short smoke test passes. It must
complete:

1. 65K x C10 in eager/no-spec mode;
2. 65K x C10 with DFlash2 and the production CUDA graph profile;
3. the complete benchmark matrix with progress-based stall detection;
4. a throughput comparison against the original six-group decode launch.
