#!/usr/bin/env bash
# One-shot health snapshot for a running llama-benchy run.
# Usage: bench_status.sh <results-dir-glob-prefix>
# Reports tracked progress, engine scheduler state, and GPU spin signature.
set -u
prefix="${1:?results dir prefix}"
dir=$(ls -td "$prefix"* 2>/dev/null | head -1)
[ -n "$dir" ] || { echo "no results dir for $prefix"; exit 2; }
python3 - "$dir/progress.jsonl" <<'PY'
import collections, json, os, sys, time
c = collections.Counter(); last = 0; err = 0
try:
    for line in open(sys.argv[1]):
        e = json.loads(line); c[e["type"]] += 1
        if e["type"] == "request_end":
            last = e.get("ts", last); err += bool(e.get("error"))
except FileNotFoundError:
    print("no progress yet"); sys.exit(0)
age = round(time.time() - last, 1) if last else None
csv = os.path.exists(os.path.join(os.path.dirname(sys.argv[1]), "results.csv"))
print(time.strftime("%H:%M:%SZ", time.gmtime()), dict(c), "errors", err, "age", age, "csv", csv)
PY
docker logs --since 1m glm53-exl3-tp4 2>&1 | grep 'Engine 000' | tail -1 | sed 's/.*Engine 000: //' | cut -c1-140
echo -n "gpu head: "; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader
