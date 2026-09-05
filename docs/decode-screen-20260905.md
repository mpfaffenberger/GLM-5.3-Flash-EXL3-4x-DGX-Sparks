# Decode screening, 2026-09-05

Exploratory only; published recipe unchanged. Evidence directory:
`results/decode-clean-20260905-135631/`.

Target/draft TP=4, GPU memory utilization 0.55, graphs off, decode
concurrency=6. llama-benchy 0.4.0, depth=0, PP2048/TG128, C1/C2,
three repetitions with normal warmup. Sequential order: k7, k3, k11.
Each screen passed 9 requests / 4 result rows and post-run API health.
The existing GPU reset procedure ran between candidates and after the last.

| Draft tokens | C1 tok/s/request | C2 tok/s/request |
|---|---:|---:|
| 7 | 47.14 | 49.57 |
| 3 | 46.88 | 43.43 |
| 11 | 42.69 | 43.59 |

Neither candidate merits promotion. Small sample, appreciable variability,
fixed run order: these numbers are screening evidence, not significance
claims. No long-context qualification or TP2 comparison was performed here.

## Restart memory

The preceding sweep (`results/decode-screen-20260905-130013/`) completed
k7 at depths 0/32K, but k3 failed during startup: CUDA reported 39.35 GiB
free versus the unchanged 66.89 GiB requested budget. After teardown the
head had about 71 GiB used, 50 GiB available, no containers or compute
processes, and only persistence-daemon device holders. Workers had about
117 GiB available. Existing module-reload + cache-drop recovery restored
the head to 117 GiB available. This implicates retained driver state, but
the combined procedure does not isolate driver memory from page cache.
This is recovered *idle* memory, not proven additional safe live KV capacity.

## Continued experiments

All screening rows below use depth=0, PP2048/TG128, C1/C2 and three
repetitions unless marked otherwise. No candidate promoted.

| Trial | C1 | C2 | Result |
|---|---:|---:|---|
| Requested draft TP1 | 41.40 | 49.13 | topology not verified; see below |
| Decode groups 8 | 45.11 | 49.74 | no useful gain |
| Decode groups 4 | 43.23 | 41.51 | reject |
| No speculation | 18.81 | 18.30 | 2.743M KV tokens; too slow |
| FP8 draft KV + builder fix | 47.54 | 47.91 | 2.078M KV tokens; no capacity win |
| AllReduce Tree | 50.63 | 47.83 | initial hint did not reproduce |
| Default NCCL, nine repetitions | 47.28 | 48.78 | confirmation baseline |
| AllReduce Tree, nine repetitions | 45.02 | 48.94 | reject |
| Native MTP2 | 37.60 | 36.59 | 2.385M KV tokens; too slow |
| Standalone draft block64 | 48.45 | 49.36 | 2.029M KV tokens; no capacity gain |
| Standalone draft block256 | 51.31 | 49.92 | initial hint did not reproduce |
| Standalone block256, nine repetitions | 46.56 | 49.34 | reject as speed optimization |

Successful three-repeat screens each passed 9 requests, four CSV rows,
and post-run health; nine-repeat screens passed 27 requests each.
Server logs and CSVs live under descriptive `results/` directories.
The fixed-order noisy workload is suitable for rejection, not a statistical
claim of small gains. The full matrix has not been rerun.

### Configuration traps

- The DFlash proposer inherits the target parallel config when constructing
  the draft VllmConfig; its override only changes attention configuration.
  Accepting `draft_tensor_parallel_size=1` does not prove effective TP1.
- `NCCL_ALGO=Tree` breaks AllGather (no available algorithm). The functioning
  scoped setting was `NCCL_ALGO=allreduce:Tree`.
- FP8 draft KV initially failed because the FlashInfer metadata builder
  received target `fp8_ds_mla`. `prepare_draft_kv_trial.py` generates an
  isolated read-only mount patch selecting the group's actual layer dtype
  for its builder. The base image and published recipe are unchanged.
- The successful FP8 draft trial logs padded slot-sharing: draft block=64,
  MLA page=1,175,552 bytes, draft bytes/token=512. Smaller draft elements
  therefore do not automatically increase shared-pool capacity. This is
  not a qualified cache-layout fix or output-quality evaluation.
- Removing draft page padding selects the allocator's existing standalone
  tensor path. Block64 still charges draft block IDs against the shared
  pool and adds per-block storage. Block256 reduces window block demand
  but increases per-block storage; capacity was only 2.109M tokens.
  Startup's idempotent drafter patcher rejects modified source, so these
  isolated mounted trials explicitly skip that one patcher. Defaults do not.

### Profile and GEMM investigations

One warmed 128-token completion was profiled with PyTorch on all ranks.
Rank0 table: EXL3 MoE ~36%, matmul ~36%, NCCL AllReduce ~19% of CUDA time.
Do not sum operator and kernel rows: they describe the same work twice.
Profiling overhead makes this unsuitable as a throughput result.
Tables: `results/decode-profile-20260905/`; full gzipped traces remain in
each host's `~/.cache/vllm-glm53-exl3-tp4/decode-profile/`.

Kernel events linked to `aten::mm` by external ID identify the largest
shape as [8,4096] x [4096,6416], about 429ms across the capture.
`bench_small_gemm.py` tests an isolated split-K Triton kernel, weight-layout
changes, N padding, and cuBLASLt preference. Split-K lost across tested
shapes; padding and cuBLASLt showed no compelling gain. Contiguous B helped
one smaller shape but hurt the dominant shape and changed numerical
results. No serving GEMM substitution was installed.

Keep shared-expert stream disabled and memory at 0.55 until separately
qualified. Any replacement still requires correctness tests, 65K/C10,
100K/C10 and the full matrix. Next deeper work: cache pooling/layout and
EXL3/linear-kernel profiling rather than repeating the rejected knobs.
