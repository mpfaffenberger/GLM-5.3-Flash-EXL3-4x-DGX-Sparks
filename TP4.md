# Four-node DGX Spark trial

This branch adds an isolated TP=4 launcher for four GB10 nodes. Mia's original
two-node launcher remains intact so the upstream recipe stays reviewable.

## Pinned inputs

- EXL3 target: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- target revision: `25a44fdbf16862a46b7cc9921142c6c81350af2f`
- DFlash2 draft: `incoai/GLM-5.3-Flash-DFlash2`
- draft revision: `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- runtime digest:
  `ghcr.io/mpfaffenberger/glm-5.3-flash-2x-dgx-sparks@sha256:03161bb433140860c6fbe9505de522f819630215d8eca9a8ba2c73652706dd96`

Default topology:

| Rank | Address | RoCE GID |
|---:|---|---:|
| 0 | `10.0.0.46` | 3 |
| 1 | `10.0.0.13` | 5 |
| 2 | `10.0.0.150` | 3 |
| 3 | `10.0.0.246` | 3 |

All ranks use `enp1s0f1np1` / `rocep1s0f1`. The launcher verifies the image,
target snapshot, draft snapshot, and each GID before touching the cluster.

## Validated profile

- target TP=4;
- DFlash2 draft TP=4 and seven draft tokens;
- `torch.compile` on, CUDA graphs **off** (`CUDAGRAPH_MODE=none`). Piecewise
  graphs are ~15% faster at C=1 but never survived the full long-context
  matrix; opt in with `CUDAGRAPH_MODE=piecewise` at your own risk;
- fused EXL3 MoE with six groups for both prefill and decode
  (`EXL3_MOE_CONCURRENCY=6 EXL3_MOE_DECODE_CONCURRENCY=6`);
- FP8 target KV cache;
- max model length 1,000,000;
- max sequences 10;
- max batched tokens 2,048;
- GPU memory utilization 0.55. Trials at 0.75 and above reached graph capture
  but exhausted practical GB10 unified-memory headroom under sustained
  long-context load at the driver layer, and 0.60 still produced a rank-local
  `CUBLAS_STATUS_INTERNAL_ERROR` after a 100K soak;
- native vLLM multiprocessing; no Ray.

Use `SPEC_METHOD=mtp MTP_TOKENS=2` or `SPEC_METHOD=none` for controlled
comparisons. These are experiments, not support claims.

## Launch

Each node must have both pinned snapshots in its normal Hugging Face cache.
Download once and synchronize over the cluster fabric with:

```bash
./scripts/tp4_sync_weights.sh
```

Then, from rank 0:

```bash
./scripts/tp4_cluster.sh start
./scripts/tp4_cluster.sh status
docker logs -f glm53-exl3-tp4
```

Stop all four ranks with:

```bash
./scripts/tp4_cluster.sh stop
```

## Benchmark

The benchmark wrapper fails closed on HTTP errors, incomplete request streams,
wrong result cardinality, or an unhealthy post-run API:

```bash
# Quick C=1 gate
DEPTHS='0' CONCURRENCIES='1' RUNS=3 \
  ./scripts/bench_llama_benchy_tp4.sh

# Full matrix through 65,535 tokens
./scripts/bench_llama_benchy_tp4.sh
```

Do not publish a partial CSV as a result. Distributed systems already produce
enough fiction without our help.

## Validated full matrix (2026-09-04)

`results/exl3-barrier-v2-mem055-full-20260904-012121/` — 702/702 requests,
zero errors, depths 0 through 100,000 at concurrency 1, 2, 5, 10, run from a
clean reset with the validated profile above. llama-benchy PP=2048 / TG=128.

| Cell | Prefill tok/s | Decode tok/s per request | TTFT |
|---|---:|---:|---:|
| C=1, no context | **959** | **44.7** | 2.4 s |
| C=1 @ 32K | 849 | 48.1 | 2.4 s |
| C=1 @ 65K | 705 | 43.6 | 2.9 s |
| C=1 @ 100K | 626 | 33.7 | 3.3 s |
| C=10, no context | 490 aggregate | 46.7 | 23 s |
| C=10 @ 65K | 36 aggregate | 43.0 | 312 s |
| C=10 @ 100K | 25 aggregate | 42.4 | 440 s |

Per-request decode holds 42–48 tok/s across concurrency thanks to DFlash.
The C=10 long-context prefill cells are TTFT-bound: ten 65–100K requests
queue through 2,048-token batches behind `max-num-seqs=10`. They are stability
cells, not a throughput target. The full diagnosis of what had to be fixed to
get here is in `docs/exl3-moe-wedge.md`.

## First TP=4 gates (2026-08-30, superseded)

The first successful API-ready run used the initial profile above. CUDA graphs
were active and the server remained healthy after each gate.

| Gate | Median decode | Acceptance | TTFT median |
|---|---:|---:|---:|
| Structured count, 5 × 400 | **105.37 tok/s** | 0.959 / 6.712 per step | 0.305 s |
| Prose hash-map, 5 × 400 | **42.02 tok/s** | 0.314 / 2.200 per step | 0.421 s |
| llama-benchy C=1, TG=128 | **51.55 tok/s mean** | mixed workload | — |
| llama-benchy C=1, PP=2048 | **1,142.36 tok/s mean** | — | — |

For comparison, Mia's published TP=2 medians on the same structured/prose
microbench protocol are 65.1 and 27.1 tok/s. The TP=4 trial is roughly 62% and
55% faster respectively. This comparison is promising, but it is not a full
long-context stability result.

The first 0.87/0.86 memory-utilization attempts also exposed a GB10 unified-
memory trap: vLLM's nominal graph/cache accounting passed, then the NVIDIA
driver emitted `NV_ERR_NO_MEMORY` during graph capture. A later sustained 65K
x C10 run showed that 0.75 still left only about 9 GiB available, entered swap,
and hit the same driver allocation failure. At 0.60 a warm engine that had just
completed a 100K x C10 soak lost one rank to `CUBLAS_STATUS_INTERNAL_ERROR`
with no Xid logged. Utilization 0.55 provides about 2.1 million KV tokens—still
2.1x one 1M-token request—and is the setting under which the full matrix
through 100K x C10 passed 702/702 (`results/exl3-barrier-v2-mem055-full-*`).

An initial `max-num-seqs=4` full-matrix attempt wedged on the third 4K × C=5
batch after two requests emitted their first token. All ranks remained at 96%
GPU utilization inside `apply_exl3_fused_moe`. This is the same dangerous
admit-a-waiting-request-while-others-decode shape seen in the FP8 profile.
Because the measured TP=4 KV capacity is far above the benchmark requirement,
the profile now admits all benchmark clients together with `max-num-seqs=10`
rather than manufacturing that avoidable scheduler transition.
The exact C=10 graph shape is included explicitly in
`--cudagraph-capture-sizes`; omitting it caused the uncaptured startup warmup
for that shape to diverge across TP ranks.
