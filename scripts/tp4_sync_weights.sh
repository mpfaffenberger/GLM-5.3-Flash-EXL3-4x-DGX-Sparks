#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58}
MODEL=${MODEL:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}
REVISION=${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}
MODEL_CACHE=${MODEL_CACHE:-models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw}
DFLASH_CACHE=${DFLASH_CACHE:-models--incoai--GLM-5.3-Flash-DFlash2}
DFLASH_REVISION=${DFLASH_REVISION:-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-120}
WORKERS=(10.0.0.13 10.0.0.150 10.0.0.246)
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e HF_HOME=/cache \
    -v "$HF_HOME:/cache" --entrypoint hf "$IMAGE" \
    download "$MODEL" --revision "$REVISION"

snapshot="$HF_HOME/hub/$MODEL_CACHE/snapshots/$REVISION"
count=$(find "$snapshot" -name '*.safetensors' -type l -o -name '*.safetensors' -type f | wc -l)
[[ "$count" == "$EXPECTED_SHARDS" ]] || {
    echo "incomplete target snapshot: $count/$EXPECTED_SHARDS shards" >&2
    exit 1
}

for item in "$MODEL_CACHE:$REVISION" "$DFLASH_CACHE:$DFLASH_REVISION"; do
    cache=${item%%:*}
    revision=${item#*:}
    source="$HF_HOME/hub/$cache"
    [[ -d "$source" ]] || { echo "missing cache: $source" >&2; exit 1; }
    pids=()
    for ip in "${WORKERS[@]}"; do
        remote="$HOME/.cache/huggingface/hub/$cache"
        if ssh "$ip" "test -f '$remote/snapshots/$revision/config.json'"; then
            echo "$ip already has $cache@$revision"
            continue
        fi
        ssh "$ip" "mkdir -p '$remote'"
        rsync -rlpt --partial --info=stats2 "$source/" "$ip:$remote/" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { echo "weight synchronization failed" >&2; exit 1; }
    for ip in "${WORKERS[@]}"; do
        ssh "$ip" "test -f '$HOME/.cache/huggingface/hub/$cache/snapshots/$revision/config.json'" \
            || { echo "snapshot validation failed on $ip" >&2; exit 1; }
    done
done

echo "pinned EXL3 and DFlash2 snapshots synchronized to all TP4 ranks"
