# Four-node DGX Spark trial

This branch adds an isolated TP=4 launcher for four GB10 nodes. Mia's original
two-node launcher remains intact so the upstream recipe stays reviewable.

## Pinned inputs

- EXL3 target: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- target revision: `25a44fdbf16862a46b7cc9921142c6c81350af2f`
- DFlash2 draft: `incoai/GLM-5.3-Flash-DFlash2`
- draft revision: `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- runtime digest:
  `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`

Default topology:

| Rank | Address | RoCE GID |
|---:|---|---:|
| 0 | `10.0.0.46` | 3 |
| 1 | `10.0.0.13` | 5 |
| 2 | `10.0.0.150` | 3 |
| 3 | `10.0.0.246` | 3 |

All ranks use `enp1s0f1np1` / `rocep1s0f1`. The launcher verifies the image,
target snapshot, draft snapshot, and each GID before touching the cluster.

## Initial trial profile

- target TP=4;
- DFlash2 draft TP=4 and seven draft tokens;
- CUDA graphs enabled;
- FP8 target KV cache;
- max model length 1,000,000;
- max sequences 4;
- max batched tokens 2,048;
- GPU memory utilization 0.87;
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
