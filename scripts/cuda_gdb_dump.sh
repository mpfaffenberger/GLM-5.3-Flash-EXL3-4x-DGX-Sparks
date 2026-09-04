#!/usr/bin/env bash
# Attach cuda-gdb to the TP worker inside the running container on <host> and
# dump GPU kernel/warp state. Diagnostic only; requires an image with
# cuda-gdb-13-0 and a container started with --cap-add SYS_PTRACE.
# Usage: cuda_gdb_dump.sh <host|local> <output-file>
set -uo pipefail
host=${1:?host}
out=${2:?output file}
name=${CONTAINER_NAME:-glm53-exl3-tp4}
gdb=/usr/local/cuda-13.0/bin/cuda-gdb
cmds=/tmp/cuda-gdb-dump.cmds

run() { if [[ $host == local ]]; then "$@"; else ssh -o BatchMode=yes "$host" "$@"; fi; }

pid=$(run docker exec "$name" pgrep -f 'VLLM::Worker' | head -1)
[[ -n $pid ]] || { echo "no worker pid on $host" >&2; exit 1; }

# Command file avoids quoting -ex arguments through ssh + docker exec.
run docker exec -i "$name" tee "$cmds" >/dev/null <<'GDB'
set pagination off
set confirm off
info cuda devices
info cuda kernels
info cuda blocks
info cuda warps
bt 12
x/6i $pc
detach
GDB

run timeout 600 docker exec "$name" "$gdb" -batch -x "$cmds" -p "$pid" >"$out" 2>&1
echo "wrote $out ($(wc -l <"$out") lines)"
