# Prefill experiments, 2026-09-06

Exploratory, not a replacement recipe. TP4, DFlash7, memory utilization
0.55, graphs NONE, C1/C2, llama-benchy 0.4.0, PP8192/TG128, depth zero
unless indicated. Normal warmup and reset between trials.

| Trial | repetitions | PP C1 | PP C2 | decode C1 | decode C2 |
|---|---:|---:|---:|---:|---:|
| Budget2048, rows128 | 3 | 1180.98 | 681.41 | 47.38 | 45.18 |
| Budget4096 | 3 | 1165.68 | 554.77 | 45.01 | 49.85 |
| Budget8192 | 3 | 1233.05 | 675.08 | 44.69 | 50.53 |
| Requested groups8, effective6 | 3 | 1195.50 | 699.30 | 49.96 | 49.68 |
| Rows256 | 3 | 1285.54 | 748.94 | 49.01 | 54.04 |
| Confirmation rows128 | 9 | 1193.75 | 691.60 | 52.41 | 47.55 |
| Confirmation rows256 | 9 | 1279.02 | 741.91 | 48.99 | 49.60 |
| Prefill256/decode128 | 9 | 1287.47 | 743.26 | 44.08 | 47.90 |
| Prefill512/decode128 | 3 | 1244.74 | 731.34 | 52.33 | 51.05 |

All values are tok/s/request. Three-repeat screens completed nine requests;
nine-repeat screens completed 27, each with four CSV rows and API health.
Fixed run order, variable decode acceptance, and large C2 prefill variance
limit causal conclusions. The 256-row prefill benefit reproduced at C1:
TTFT 6.866s baseline versus 6.408s candidate. Decode non-regression is NOT
established. No 65K/C10, 100K/C10, full matrix, or quality qualification yet.

## Findings

- Larger scheduler token budgets did not give a compelling improvement.
- Requested MoE concurrency8 was clamped to6 by the extension's device limit;
  server logs confirm6. This was not an effective configuration change.
- Baseline prefill statistics show >90% of recorded calls had an expert
  exceeding the 128-row scratch capacity, invoking a Python fallback.
- Increasing scratch rows to256 improved prefill by approximately7% in
  nine-repeat comparisons, but C1 decode averaged lower.
- Optional `EXL3_TEMP_ROWS_DECODE` separates decode scratch allocation from
  prefill. Defaults inherit prefill rows, preserving previous behavior.
  Setting256/128 retained the prefill benefit, not a decode recovery.
- Editing the active shell runner's logging caused a post-recovery parse
  error after the rows128 confirmation. The benchmark completed and reset
  logs verify recovery; the candidate was then launched separately. Do not
  edit shell scripts while they are executing.

## 32K depth

256/128 completed 18 requests and eight rows at depth32768, PP8192/TG128.
C1 context ingestion1349.53, subsequent PP1211.92, subsequent decode48.40.
C2 context ingestion940.93, subsequent PP719.09, subsequent decode45.97.
These distinguish context ingestion from prefix-cached incremental prompts;
they are not interchangeable measurements.

Matched128/128 baseline also passed18 requests/eight rows. C1 context
ingestion1231.38 and subsequent PP1122.33, versus1349.53/1211.92 for256/128
(+9.6%/+8.0%). C2 baseline context875.53 and subsequent PP667.23, versus
940.93/719.09 (+7.5%/+7.8%). Decode remains mixed: subsequent C1
45.53→48.40, C2 47.05→45.97; context-generation C1 42.75→39.22,
C2 43.63→41.46. Candidate ran first here, reversing the zero-depth
confirmation order. This strengthens the prefill lead but is not full
qualification or evidence of a decode improvement.

Evidence is under `results/prefill-*`. Published recipe remains unchanged.
