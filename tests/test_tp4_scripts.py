from pathlib import Path


ROOT = Path(__file__).parents[1]
INNER = (ROOT / "scripts/tp4_inner.sh").read_text()
CLUSTER = (ROOT / "scripts/tp4_cluster.sh").read_text()


def test_tp4_rank_contract_is_consistent() -> None:
    assert "--tensor-parallel-size 4" in INNER
    assert "--nnodes 4" in INNER
    assert '--node-rank "$NODE_RANK"' in INNER
    assert '[[ "$NODE_RANK" != 0 ]]' in INNER
    assert "10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246" in CLUSTER


def test_tp4_inputs_are_immutable() -> None:
    assert "@sha256:9bb1557a4234fce" in CLUSTER
    assert "25a44fdbf16862a46b7cc9921142c6c81350af2f" in CLUSTER
    assert "dc77ff1c99eeb2df044ee3d4f0094eb033fee410" in CLUSTER


def test_tp4_dflash_and_graph_defaults() -> None:
    assert 'DFLASH_DRAFT_TP:-4' in CLUSTER
    assert 'SPEC_METHOD:-dflash' in INNER
    assert "--cudagraph-capture-sizes 1 2 4 8 16 24 32" in INNER
    assert '--enforce-eager' in INNER


def test_tp4_preflight_happens_before_stop() -> None:
    preflight = CLUSTER.index('echo "preflighting immutable image')
    stop = CLUSTER.index("\nstop_cluster\n", preflight)
    assert preflight < stop
