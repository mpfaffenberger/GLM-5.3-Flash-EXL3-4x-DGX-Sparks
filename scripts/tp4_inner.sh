#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?NODE_RANK is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${HEAD_IP:?HEAD_IP is required}"

say() { printf '[glm53-exl3-tp4 rank=%s] %s\n' "$NODE_RANK" "$*"; }

# These are idempotent. Keep runtime and image-built behavior aligned while
# Mia's image remains an external dependency.
for patcher in \
    patch_glm_video_placeholders.py \
    patch_suppress_stops_in_reasoning.py \
    patch_scheduler_decode_floor.py \
    patch_glm5_drafter_group.py \
    patch_hybrid_prefix_hit.py \
    patch_xgrammar_termination.py \
    patch_kpool_tail_slotmap.py; do
    [[ -f "/opt/glm53/$patcher" ]] && python3 "/opt/glm53/$patcher"
done

[[ -f "$MODEL_DIR/config.json" ]] || {
    say "FATAL: model config missing at $MODEL_DIR"
    exit 2
}

args=(
    --served-model-name "${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
    --host 0.0.0.0
    --port "${PORT:-8888}"
    --tensor-parallel-size 4
    --nnodes 4
    --node-rank "$NODE_RANK"
    --master-addr "$HEAD_IP"
    --master-port "${MASTER_PORT:-29521}"
    --distributed-executor-backend mp
    --quantization exl3
    --max-model-len "${MAX_MODEL_LEN:-1000000}"
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.87}"
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --kv-cache-dtype fp8
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --chat-template /opt/glm53/chat_template.jinja
    --limit-mm-per-prompt '{"image":4,"video":1}'
    --skip-mm-profiling
)

if [[ "$NODE_RANK" != 0 ]]; then
    args+=(--headless)
fi

if [[ "${ENFORCE_EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
else
    args+=(--cudagraph-capture-sizes 1 2 4 8 16 24 32)
fi

case "${SPEC_METHOD:-dflash}" in
    dflash)
        : "${DFLASH_MODEL_DIR:?DFLASH_MODEL_DIR is required for DFlash2}"
        spec=$(python3 -S - <<'PY'
import json
import os

print(json.dumps({
    "method": "dflash",
    "model": os.environ["DFLASH_MODEL_DIR"],
    "num_speculative_tokens": int(os.environ.get("DFLASH_TOKENS", "7")),
    "draft_tensor_parallel_size": int(os.environ.get("DFLASH_DRAFT_TP", "4")),
    "kv_cache_dtype": "auto",
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "standard",
}, separators=(",", ":")))
PY
)
        args+=(--speculative-config "$spec")
        ;;
    mtp)
        args+=(--speculative-config \
            "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-2}}")
        ;;
    none) ;;
    *) say "FATAL: unknown SPEC_METHOD=${SPEC_METHOD}"; exit 2 ;;
esac

say "starting TP=4 node (spec=${SPEC_METHOD:-dflash}, graphs=$((1 - ${ENFORCE_EAGER:-0})))"
exec vllm serve "$MODEL_DIR" "${args[@]}"
