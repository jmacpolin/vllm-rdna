# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for gfx10x oneshot all-reduce helpers.

Loads ``rdna_all_reduce.py`` by path so collection does not import
``vllm.distributed`` (heavy). Does not launch the HIP kernel. Guards the
T44b abort-record decode, the wedge-marker path, and that VLLM_RDNA_AR
stays default-off.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "vllm" / "distributed" / "device_communicators" / "rdna_all_reduce.py"


def _load_rdna_ar(monkeypatch: pytest.MonkeyPatch, cache_root: Path):
    """Import rdna_all_reduce.py with stub vllm.logger / platforms / envs."""
    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
    platforms_mod = types.ModuleType("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace()  # type: ignore[attr-defined]
    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_CACHE_ROOT = str(cache_root)  # type: ignore[attr-defined]
    vllm_mod.envs = envs_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_mod)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_mod)

    spec = importlib.util.spec_from_file_location("rdna_all_reduce", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pack(phase: int, peer: int, ms: int, seq: int) -> int:
    """Match rdna_ar_ms_code in csrc/rocm/rdna_allreduce.cuh."""
    return (
        1
        | ((phase & 0xF) << 8)
        | ((peer & 0xF) << 12)
        | (min(ms // 16, 0xFFFF) << 16)
        | ((seq & 0xFFFFFFFF) << 32)
    )


def test_describe_abort_grid_barrier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=1, peer=0, ms=120_000, seq=7)
    msg = rdna_ar.describe_abort(code, rank=2)
    assert "rank 2" in msg
    assert "collective #7" in msg
    assert "~120000 ms" in msg
    assert "grid barrier" in msg


def test_describe_abort_long_wait_saturates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """16 ms units: the 120 s default fits; ~17 min and beyond saturate."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    msg = rdna_ar.describe_abort(_pack(phase=2, peer=1, ms=2_000, seq=4), 0)
    assert "~2000 ms" in msg
    sat = rdna_ar.describe_abort(_pack(phase=2, peer=1, ms=1 << 40, seq=4), 0)
    assert f"~{0xFFFF * 16} ms" in sat


def test_describe_abort_peer_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=2, peer=3, ms=1_200, seq=9)
    msg = rdna_ar.describe_abort(code, rank=0)
    assert "peer rank 3" in msg
    assert "collective #9" in msg
    assert "posted P2P write from GPU 3" in msg


@pytest.mark.parametrize(
    "env, gate, dropped",
    [
        ({}, 64, None),
        ({"VLLM_RDNA_AR_MAX_KB": "32"}, 32, None),
        ({"VLLM_RDNA_AR_ALGO": "oneshot"}, 64, None),
        ({"VLLM_RDNA_AR_MAX_KB": "20480"}, 64, "VLLM_RDNA_AR_MAX_KB=20480"),
        ({"VLLM_RDNA_AR_ALGO": "twoshot"}, 64, "VLLM_RDNA_AR_ALGO=twoshot"),
        ({"VLLM_RDNA_AR_ALGO": "2stage"}, 64, "VLLM_RDNA_AR_ALGO=2stage"),
        (
            {"VLLM_RDNA_AR_MAX_KB": "256", "VLLM_RDNA_AR_ONESHOT_KB": "128"},
            128,
            "VLLM_RDNA_AR_MAX_KB=256",
        ),
    ],
)
def test_twoshot_cannot_be_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env, gate, dropped
):
    """Two-shot was removed: any request for it keeps one-shot up to the gate
    (RCCL above) and says so once."""
    for name in ("VLLM_RDNA_AR_MAX_KB", "VLLM_RDNA_AR_ONESHOT_KB", "VLLM_RDNA_AR_ALGO"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    got_gate, note = rdna_ar.resolve_gate_kb()
    assert got_gate == gate
    if dropped is None:
        assert note is None
    else:
        assert dropped in note
        assert f"one-shot up to {gate} KiB and RCCL above" in note


def test_twoshot_kernel_is_gone():
    src = (_ROOT / "csrc/rocm/rdna_allreduce.cu").read_text()
    hdr = (_ROOT / "csrc/rocm/rdna_allreduce.cuh").read_text()
    assert "twoshot" not in src
    assert "rdna_ar_twoshot" not in hdr
    # The eligibility check caps at the one-shot gate, so a larger tensor is
    # never handed to the one-shot kernel either.
    assert "std::min(g.max_bytes, g.oneshot_max)" in src


def test_marker_path_uses_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.marker_path() == str(tmp_path / "rdna_ar_wedged")


def test_rdna_ar_defaults_off():
    comm = (
        _ROOT / "vllm/distributed/device_communicators/cuda_communicator.py"
    ).read_text()
    envs = (_ROOT / "vllm/envs.py").read_text()
    assert 'os.getenv("VLLM_RDNA_AR", "0")' in comm
    assert 'os.getenv("VLLM_RDNA_AR", "0")' in envs


def test_rdna_ar_check_noop_without_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar._live == []
    rdna_ar.rdna_ar_check()  # must not raise when no instance is active


def test_rdna_ar_check_covers_every_live_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A wedge on the second instance (e.g. the EP group) fails the step too;
    an already-disabled instance is skipped."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def __init__(self, code: int) -> None:
            self.code = code
            self.calls = 0

        def rdna_ar_timeout_info(self, handle: int) -> int:
            self.calls += 1
            return self.code

    def make(handle: int, code: int, disabled: bool = False):
        inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
        inst.disabled = disabled
        inst.handle = handle
        inst.rank = 2
        inst.world_size = 4
        inst._ops = _Ops(code)
        return inst

    tp, gone, ep = (
        make(0, 0),
        make(1, _pack(2, 3, 120_000, 5), disabled=True),
        make(2, _pack(2, 1, 120_000, 9)),
    )
    rdna_ar._live.extend([tp, gone, ep])
    with pytest.raises(RuntimeError, match="peer rank 1"):
        rdna_ar.rdna_ar_check()
    assert tp._ops.calls == 1 and gone._ops.calls == 0 and ep.disabled
    rdna_ar.rdna_ar_check()  # the wedged instance is disabled now
    assert tp._ops.calls == 2


def test_check_writes_marker_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return _pack(phase=2, peer=1, ms=120_000, seq=3)

    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.disabled = False
    inst.handle = 0
    inst.rank = 0
    inst.world_size = 4
    inst._ops = _Ops()
    with pytest.raises(RuntimeError, match="rdna_ar wedged"):
        inst.check()
    marker = tmp_path / "rdna_ar_wedged"
    assert marker.is_file()
    assert "peer rank 1" in marker.read_text(encoding="utf-8")
    assert inst.disabled is True


def test_check_logs_late_peer_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_slow_info(self, handle: int) -> int:
            return _pack(phase=2, peer=3, ms=5_008, seq=11)

        def rdna_ar_timeout_info(self, handle: int) -> int:
            return 0

    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.disabled = False
    inst.handle = 0
    inst.rank = 1
    inst.world_size = 4
    inst._ops = _Ops()
    inst.check()  # a late peer is logged, not fatal
    assert inst.disabled is False
    assert not (tmp_path / "rdna_ar_wedged").exists()
    msg = rdna_ar.describe_late(_pack(phase=2, peer=3, ms=5_008, seq=11), 1)
    assert msg == "rank 1 waited ~5008 ms for peer rank 3 at collective #11"
