#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HEAD_IP=${HEAD_IP:-10.0.0.46}
NODES=(10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246)
IFS=' ' read -r -a GIDS <<< "${GIDS:-3 5 3 3}"
SOCKET_IF=${SOCKET_IF:-enp1s0f1np1}
HCA=${HCA:-rocep1s0f1}
NAME=${CONTAINER_NAME:-glm53-exl3-tp4}
IMAGE=${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58}
MODEL_REVISION=${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}
DFLASH_REVISION=${DFLASH_REVISION:-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}
MODEL_CACHE=models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw
DFLASH_CACHE=models--incoai--GLM-5.3-Flash-DFlash2
ACTION=${1:-start}

run_on() {
    local ip=$1; shift
    if [[ "$ip" == "$HEAD_IP" ]]; then "$@"; else ssh -o BatchMode=yes "$ip" "$@"; fi
}

stop_cluster() {
    local ip
    for ip in "${NODES[@]}"; do
        run_on "$ip" docker rm -f "$NAME" >/dev/null 2>&1 &
    done
    wait || true
}

if [[ "$ACTION" == stop ]]; then
    stop_cluster
    echo "EXL3 TP4 cluster stopped"
    exit 0
fi

if [[ "$ACTION" == status ]]; then
    for ip in "${NODES[@]}"; do
        printf '%s ' "$ip"
        run_on "$ip" docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null \
            || echo stopped
    done
    curl -sS -o /dev/null -w 'API %{http_code}\n' --max-time 5 \
        "http://127.0.0.1:${PORT:-8888}/health" || true
    exit 0
fi

[[ "$ACTION" == start ]] || { echo "usage: $0 [start|stop|status]" >&2; exit 2; }
[[ ${#GIDS[@]} == 4 ]] || { echo "GIDS must contain four values" >&2; exit 2; }

model_rel="hub/$MODEL_CACHE/snapshots/$MODEL_REVISION"
dflash_rel="hub/$DFLASH_CACHE/snapshots/$DFLASH_REVISION"

echo "preflighting immutable image, snapshots, RoCE, and disk on all four nodes"
for rank in "${!NODES[@]}"; do
    ip=${NODES[$rank]}
    run_on "$ip" test -f "$HOME/.cache/huggingface/$model_rel/config.json" || {
        echo "missing EXL3 snapshot on $ip" >&2; exit 1;
    }
    if [[ "${SPEC_METHOD:-dflash}" == dflash ]]; then
        run_on "$ip" test -f "$HOME/.cache/huggingface/$dflash_rel/config.json" || {
            echo "missing DFlash2 snapshot on $ip" >&2; exit 1;
        }
    fi
    run_on "$ip" docker image inspect "$IMAGE" >/dev/null
    gid=$(run_on "$ip" cat "/sys/class/infiniband/$HCA/ports/1/gids/${GIDS[$rank]}")
    [[ "$gid" != 0000:0000:0000:0000:0000:0000:0000:0000 ]] || {
        echo "empty RoCE GID on $ip index ${GIDS[$rank]}" >&2; exit 1;
    }
    python3 - "$gid" "$ip" <<'PY' || {
import ipaddress
import sys

gid = ipaddress.IPv6Address(sys.argv[1])
expected = ipaddress.IPv4Address(sys.argv[2])
if gid.ipv4_mapped != expected:
    raise SystemExit(f"GID {gid} does not map to node address {expected}")
PY
        echo "wrong RoCE GID on $ip index ${GIDS[$rank]}" >&2
        exit 1
    }
done

stop_cluster
inner=/tmp/glm53-exl3-tp4-inner.sh
exl3_overlay=/tmp/glm53-exl3-tp4-exl3.py
for rank in 1 2 3; do
    scp -q "$ROOT/scripts/tp4_inner.sh" "${NODES[$rank]}:$inner"
    scp -q "$ROOT/overlay/exl3.py" "${NODES[$rank]}:$exl3_overlay"
done
cp "$ROOT/scripts/tp4_inner.sh" "$inner"
cp "$ROOT/overlay/exl3.py" "$exl3_overlay"

common_env=(
    -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="${MASTER_PORT:-29521}"
    -e PORT="${PORT:-8888}" -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
    -e MODEL_DIR="/root/.cache/huggingface/$model_rel"
    -e DFLASH_MODEL_DIR="/root/.cache/huggingface/$dflash_rel"
    -e SPEC_METHOD="${SPEC_METHOD:-dflash}" -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
    -e DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-4}" -e MTP_TOKENS="${MTP_TOKENS:-2}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
-e MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
    -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    -e ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
    -e EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
    -e EXL3_FUSED_MOE_DECODE="${EXL3_FUSED_MOE_DECODE:-1}"
    -e EXL3_MOE_CONCURRENCY="${EXL3_MOE_CONCURRENCY:-6}"
    -e EXL3_MOE_DECODE_CONCURRENCY="${EXL3_MOE_DECODE_CONCURRENCY:-1}"
    -e EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
    -e EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}"
    -e GLM53_SUPPRESS_STOPS_IN_REASONING=1 -e GLM53_MIXED_PREFILL_CHUNK=skip
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0
    # Match the already-proven four-node fabric profile. Mia's two-node-only
    # NET_PLUGIN/MERGE_NICS settings stall communicator setup here.
    -e NCCL_CROSS_NIC=1 -e NCCL_IGNORE_CPU_AFFINITY=1
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
)

for rank in 1 2 3; do
    ip=${NODES[$rank]}
    cache="$HOME/.cache/huggingface"
    vcache="$HOME/.cache/vllm-glm53-exl3-tp4"
    remote_env=$(printf ' %q' "${common_env[@]}")
    ssh -o BatchMode=yes "$ip" "mkdir -p '$vcache/triton' '$vcache/tilelang'; docker run -d --name '$NAME' \
      --gpus all --network host --ipc=host --shm-size 64g --stop-timeout 60 \
      --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
      --ulimit nofile=1048576:1048576 \
      -v '$cache:/root/.cache/huggingface' -v '$vcache:/root/.cache/vllm' \
      -v '$vcache/triton:/root/.triton/cache' -v '$vcache/tilelang:/root/.tilelang/cache' \
      -v '$inner:/start.sh:ro' $remote_env -e NODE_RANK='$rank' \
      -v '$exl3_overlay:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro' \
      -e NCCL_SOCKET_IFNAME='$SOCKET_IF' -e GLOO_SOCKET_IFNAME='$SOCKET_IF' \
      -e TP_SOCKET_IFNAME='$SOCKET_IF' \
      -e NCCL_IB_HCA='$HCA' -e NCCL_IB_GID_INDEX='${GIDS[$rank]}' -e VLLM_HOST_IP='$ip' \
      --entrypoint bash '$IMAGE' /start.sh" >/dev/null
    echo "worker rank $rank launched on $ip"
done

vcache="$HOME/.cache/vllm-glm53-exl3-tp4"
mkdir -p "$vcache/triton" "$vcache/tilelang"
docker run -d --name "$NAME" \
    --gpus all --network host --ipc=host --shm-size 64g --stop-timeout 60 \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$vcache:/root/.cache/vllm" \
    -v "$vcache/triton:/root/.triton/cache" -v "$vcache/tilelang:/root/.tilelang/cache" \
    -v "$inner:/start.sh:ro" "${common_env[@]}" -e NODE_RANK=0 \
    -v "$exl3_overlay:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
    -e NCCL_SOCKET_IFNAME="$SOCKET_IF" -e GLOO_SOCKET_IFNAME="$SOCKET_IF" \
    -e TP_SOCKET_IFNAME="$SOCKET_IF" \
    -e NCCL_IB_HCA="$HCA" -e NCCL_IB_GID_INDEX="${GIDS[0]}" -e VLLM_HOST_IP="$HEAD_IP" \
    --entrypoint bash "$IMAGE" /start.sh >/dev/null

echo "EXL3 TP4 launch dispatched; logs: docker logs -f $NAME"
