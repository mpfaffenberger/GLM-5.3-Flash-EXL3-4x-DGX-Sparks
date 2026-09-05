#!/usr/bin/env bash
# Bounded exploratory sweep. Never promotes a profile or edits the recipe.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: screen_decode.sh NEW_RESULT_DIRECTORY}
[[ ! -e "$OUT" ]] || { echo "refusing existing output directory" >&2; exit 2; }
mkdir -p "$OUT"
exec 9>"$HOME/.glm53-decode-screen.lock"
flock -n 9 || { echo "another decode screen owns the lock" >&2; exit 2; }
export CONTAINER_NAME=glm53-decode-screen
export GPU_MEM_UTIL=0.55 CUDAGRAPH_MODE=none
export DFLASH_DRAFT_TP=${DFLASH_DRAFT_TP:-4}
export EXL3_MOE_DECODE_CONCURRENCY=${EXL3_MOE_DECODE_CONCURRENCY:-6}
for host in 10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246; do
    running=$(ssh -o BatchMode=yes "$host" 'docker ps -q')
    [[ -z "$running" ]] || { echo "occupied cluster: $host" >&2; exit 2; }
done
git -C "$ROOT" rev-parse HEAD > "$OUT/commit.txt"
capture() {
    local dir=$1
    for host in 10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246; do
        ssh -o BatchMode=yes "$host" "docker logs $CONTAINER_NAME" \
            > "$dir/$host.server.log" 2>&1 || true
        ssh -o BatchMode=yes "$host" 'cat /proc/meminfo; cat /proc/vmstat' \
            > "$dir/$host.memory.txt" || true
    done
}
cleanup() { bash "$ROOT/scripts/tp4_cluster.sh" stop; }
trap cleanup EXIT
trap 'exit 130' INT TERM
for k in ${DRAFT_LENGTHS:-7 3 11}; do
    dir="$OUT/k$k"
    mkdir "$dir"
    export DFLASH_TOKENS=$k
    printf 'DFLASH_TOKENS=%s\nGPU_MEM_UTIL=0.55\nDRAFT_TP=%s\nDECODE_GROUPS=%s\n' \
        "$k" "$DFLASH_DRAFT_TP" "$EXL3_MOE_DECODE_CONCURRENCY" > "$dir/settings.txt"
    printf 'SPEC_METHOD=%s\nDFLASH_KV_DTYPE=%s\n' "${SPEC_METHOD:-dflash}" "${DFLASH_KV_DTYPE:-auto}" >> "$dir/settings.txt"
    printf 'NCCL_ALGO=%s\nEXTRA_DOCKER_ARGS=%s\n' "${NCCL_ALGO:-auto}" "${EXTRA_DOCKER_ARGS:-}" >> "$dir/settings.txt"
    bash "$ROOT/scripts/tp4_cluster.sh" start > "$dir/launch.log" 2>&1
    ready=0
    for ((i=0; i<120; i++)); do
        state=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")
        if [[ "$state" != true ]]; then
            capture "$dir"; echo "server exited during startup k=$k"; exit 1
        fi
        if curl -fsS --max-time 2 http://127.0.0.1:8888/health >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 10
    done
    if [[ "$ready" != 1 ]]; then capture "$dir"; echo "startup failed k=$k"; exit 1; fi
    # Identical inputs, repetitions and warmup; these are screens, not gates.
    if ! DEPTHS="${SCREEN_DEPTHS:-0 32768}" CONCURRENCIES='1 2' RUNS="${SCREEN_RUNS:-3}" \
        timeout 1800 bash "$ROOT/scripts/bench_llama_benchy_tp4.sh" "$dir/bench"; then
        capture "$dir"; echo "screen failed k=$k"; exit 1
    fi
    capture "$dir"
    cleanup
    if [[ -n "${RECOVERY_SCRIPT:-}" ]]; then
        # Explicit opt-in: the supplied procedure reloads NVIDIA modules.
        # Never reset a GPU now occupied by an unrelated container.
        for host in 10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246; do
            [[ -z "$(ssh "$host" 'docker ps -q')" ]] || exit 1
        done
        GLM53_CONTAINER="$CONTAINER_NAME" bash "$RECOVERY_SCRIPT" > "$dir/recovery.log" 2>&1
    fi
    # UMA page-cache/driver state can survive container teardown. Refuse to
    # compare the next candidate without recovering the baseline headroom.
    for host in 10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246; do
        available=$(ssh -o BatchMode=yes "$host" "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo")
        if (( available < 100 * 1024 * 1024 )); then
            echo "memory recovery required on $host; refusing contaminated comparison"
            exit 1
        fi
    done
done
echo "completed screens; no candidate qualified for promotion" > "$OUT/completed.txt"
