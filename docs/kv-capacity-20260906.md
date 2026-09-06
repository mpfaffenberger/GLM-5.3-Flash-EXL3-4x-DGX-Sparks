# KV budget investigation, 2026-09-06

Published default remains 0.55. `gpu_memory_utilization` budgets memory,
not compute utilization. GB10 shares RAM between CPU and GPU.

Clean 0.60 startup reported 24.19 GiB available KV memory per rank and
2,759,018 advertised tokens, versus roughly 2.1M at0.55. During startup,
head MemAvailable was29,508,000 KiB (~28.1GiB), workers32–33GiB, and swap
usage had increased. These point-in-time observations do not isolate the
source of unaccounted memory or establish an OOM threshold. Moving0.60 to
0.85 would increase the nominal budget by another30.4GiB per node, so85%
is not a safe assumption without reducing/understanding live overhead.

## Trials

- `results/kv-mem060-c10-20260906-115727/`:
  depth0, PP8192/TG128, C10, three repetitions. Passed30 requests,
  two CSV rows and post-run health, followed by recovery.
- `results/kv-mem060-65k-c10-20260906-121242/`: depth65535, PP2048/TG128,
  C10, one repetition. Hit the runner's1800-second timeout:20 request starts,
  12 completions, no errors in completed requests. No final CSV qualification.
  Logs showed continued prefill work and deferred requests, not a confirmed
  kernel wedge. No cuBLAS/OOM error found in captured rank logs. Manual
  recovery completed after the failed screen's cleanup.

The historical0.60 failure after100K soak remains unresolved; the short
success does not supersede it. No higher budget was promoted or85% tested.

Next: investigate admission/deferred-request accounting, collect time-series
system and allocator memory, and rerun long-context testing with a suitable
bounded timeout. Include post-soak short requests and the full matrix before
promoting0.60. Do not mistake advertised token capacity for verified live
concurrency.

Runner now allows explicit SCREEN_GPU_MEM_UTIL and SCREEN_CONCURRENCIES;
defaults remain0.55 and C1/C2. Model weights, scratch defaults, speculation,
and recipe are unchanged.
