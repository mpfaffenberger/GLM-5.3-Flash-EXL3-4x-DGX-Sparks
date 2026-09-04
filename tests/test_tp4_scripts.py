from pathlib import Path


ROOT = Path(__file__).parents[1]
INNER = (ROOT / "scripts/tp4_inner.sh").read_text()
CLUSTER = (ROOT / "scripts/tp4_cluster.sh").read_text()
SYNC = (ROOT / "scripts/tp4_sync_weights.sh").read_text()
BENCH = (ROOT / "scripts/bench_llama_benchy_tp4.sh").read_text()


def test_tp4_rank_contract_is_consistent() -> None:
    assert "--tensor-parallel-size 4" in INNER
    assert "--nnodes 4" in INNER
    assert "--disable-custom-all-reduce" in INNER
    assert '--node-rank "$NODE_RANK"' in INNER
    assert '[[ "$NODE_RANK" != 0 ]]' in INNER
    assert 'HEAD_IP=${HEAD_IP:-10.0.0.46}' in CLUSTER
    assert 'WORKER_NODES:-10.0.0.13 10.0.0.150 10.0.0.246' in CLUSTER
    assert 'NODES=("$HEAD_IP" "${_workers[@]}")' in CLUSTER


def test_tp4_inputs_are_immutable() -> None:
    assert "@sha256:03161bb433140860" in CLUSTER
    assert "25a44fdbf16862a46b7cc9921142c6c81350af2f" in CLUSTER
    assert "dc77ff1c99eeb2df044ee3d4f0094eb033fee410" in CLUSTER
    assert "EXPECTED_SHARDS:-120" in SYNC
    assert "--revision \"$REVISION\"" in SYNC
    assert 'wait "$pid" || failed=1' in SYNC


def test_tp4_dflash_and_graph_defaults() -> None:
    assert 'DFLASH_DRAFT_TP:-4' in CLUSTER
    assert 'SPEC_METHOD:-dflash' in INNER
    assert "--cudagraph-capture-sizes 1 2 4 8 10 16 24 32" in INNER
    assert "CUDAGRAPH_MODE:-none" in INNER
    assert "--compilation-config" in INNER
    assert 'CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-none}"' in CLUSTER
    assert 'GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"' in CLUSTER
    assert 'GPU_MEM_UTIL:-0.55' in INNER
    assert '--enforce-eager' in INNER
    assert "NCCL_CROSS_NIC=1" in CLUSTER
    assert 'EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"' in CLUSTER
    assert "EXL3_FUSED_MOE_DECODE" in CLUSTER
    assert "EXL3_MOE_CONCURRENCY" in CLUSTER
    assert 'EXL3_MOE_DECODE_CONCURRENCY="${EXL3_MOE_DECODE_CONCURRENCY:-6}"' in CLUSTER
    assert "glm53-exl3-tp4-exl3.py" in CLUSTER
    # EXL3 cooperative kernels deadlock when vLLM overlaps shared experts on
    # its aux stream; the launcher must pin the stream off by default.
    assert 'VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"' in CLUSTER
    assert 'GEN_TOKENS=${TG:-128}' in BENCH
    assert "65535 100000" in BENCH
    assert '"${NO_WARMUP:-0}" == 1' in BENCH


def test_tp4_preflight_happens_before_stop() -> None:
    preflight = CLUSTER.index('echo "preflighting immutable image')
    stop = CLUSTER.index("\nstop_cluster\n", preflight)
    assert preflight < stop
    assert "gid.ipv4_mapped != expected" in CLUSTER
