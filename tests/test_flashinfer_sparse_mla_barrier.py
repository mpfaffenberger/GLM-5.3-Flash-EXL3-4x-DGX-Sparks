from pathlib import Path

from overlay.patch_flashinfer_sparse_mla_barrier import patch


def test_alternates_prefill_pipeline_barriers(tmp_path: Path) -> None:
    root = tmp_path / "sparse_mla_sm120"
    (root / "arch").mkdir(parents=True)
    (root / "arch/barrier.cuh").write_text(
        "template <int ID, int CNT> void bar_arrive_t();\n"
        "template <int ID, int CNT> void bar_sync_t();\n"
        "// mbarrier (SM90+) for async copy tracking\n"
    )
    (root / "prefill_kernel.cuh").write_text(
        "\n".join(
            ["bar_sync_t<1, BLOCK_THREADS>();"] * 3
            + ["bar_arrive_t<1, BLOCK_THREADS>();"] * 2
        )
    )

    assert patch(root)
    barrier = (root / "arch/barrier.cuh").read_text()
    kernel = (root / "prefill_kernel.cuh").read_text()
    assert "bar_sync_alt" in barrier
    assert "bar_arrive_alt" in barrier
    assert kernel.count("bar_sync_alt<1, 5, BLOCK_THREADS>(ti & 1);") == 3
    assert kernel.count("bar_arrive_alt<1, 5, BLOCK_THREADS>(ti & 1);") == 2
    assert not patch(root)


def test_image_applies_barrier_patch() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()
    assert "COPY overlay/patch_flashinfer_sparse_mla_barrier.py" in dockerfile
    assert "RUN python3 /opt/glm53/patch_flashinfer_sparse_mla_barrier.py" in dockerfile
    assert "RUN rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/" in dockerfile
