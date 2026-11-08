# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launchers' thermal abort (rdna_serve_guarded in rdna_launcher_common.sh).

Every sensor is a fake amdgpu hwmon tree (RDNA_HWMON_ROOT) and every server is a
small shell script, so no GPU is touched.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
COMMON = REPO / "tools/rdna/rdna_launcher_common.sh"
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
LINUX = sys.platform.startswith("linux")

pytestmark = pytest.mark.skipif(not LINUX, reason="bash, sysfs-shaped tree")


def _card(root: Path, n: int, bdf: str, temps: dict[str, int], name="amdgpu"):
    """hwmon<n> for the device ``bdf``: {label: millidegrees}."""
    dev = root / "devices" / bdf
    dev.mkdir(parents=True, exist_ok=True)
    hw = root / "class" / f"hwmon{n}"
    hw.mkdir(parents=True, exist_ok=True)
    (hw / "name").write_text(name + "\n")
    if not (hw / "device").exists():
        (hw / "device").symlink_to(dev)
    for i, (label, milli) in enumerate(temps.items(), start=1):
        (hw / f"temp{i}_input").write_text(f"{milli}\n")
        (hw / f"temp{i}_label").write_text(label + "\n")
    return hw


def _env(hwmon: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("THERMAL_")}
    env.update(RDNA_HWMON_ROOT=str(hwmon / "class"), **extra)
    return env


def _bash(script: str, env: dict[str, str], timeout: int = 60):
    return subprocess.run(
        ["bash", "-c", f'source "{COMMON}"; {script}'],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


@pytest.fixture()
def four_cards(tmp_path: Path) -> Path:
    for n, bdf in enumerate(
        ["0000:64:00.0", "0000:69:00.0", "0000:6f:00.0", "0000:72:00.0"]
    ):
        _card(tmp_path, n, bdf, {"edge": 40000, "junction": 45000, "mem": 42000})
    _card(tmp_path, 4, "coretemp.0", {"Package id 0": 99000}, name="coretemp")
    return tmp_path


def test_sensor_list_ignores_other_drivers(four_cards: Path):
    res = _bash("rdna_gpu_temps", _env(four_cards))
    lines = res.stdout.splitlines()
    assert len(lines) == 12, res.stdout
    assert "0000:6f:00.0 junction 45" in lines
    assert not any("coretemp" in ln or "Package" in ln for ln in lines)


def test_guard_gpus_narrows_the_watch(four_cards: Path):
    env = _env(four_cards, THERMAL_GUARD_GPUS="0000:6f:00.0, 72:00.0")
    res = _bash("rdna_gpu_temps", env)
    assert {ln.split()[0] for ln in res.stdout.splitlines()} == {
        "0000:6f:00.0",
        "0000:72:00.0",
    }


@pytest.mark.parametrize("milli, hot", [(89999, False), (90000, True), (91000, True)])
def test_the_limit_is_inclusive(tmp_path: Path, milli: int, hot: bool):
    _card(tmp_path, 0, "0000:64:00.0", {"edge": 40000, "junction": milli})
    res = _bash("rdna_gpu_hot 90", _env(tmp_path))
    assert (res.returncode == 0) is hot
    if hot:
        assert res.stdout.strip() == f"junction {milli // 1000}°C on 0000:64:00.0"


@pytest.mark.parametrize(
    "value, ok", [("", True), ("85", True), ("90", True), ("91", False), ("x", False)]
)
def test_the_limit_can_be_lowered_not_raised(tmp_path: Path, value: str, ok: bool):
    env = _env(tmp_path)
    if value:
        env["THERMAL_ABORT_C"] = value
    res = _bash("rdna_thermal_limit", env)
    assert (res.returncode == 0) is ok, res.stderr
    if ok:
        assert res.stdout.strip() == (value or "90")
    else:
        assert "from 1 to 90" in res.stderr


def test_the_server_status_passes_through(four_cards: Path):
    res = _bash("rdna_serve_guarded bash -c 'exit 7'", _env(four_cards))
    assert res.returncode == 7, res.stderr
    assert "serving stops at 90°C on any of 12 GPU sensors" in res.stderr


def test_refuses_without_sensors(tmp_path: Path):
    (tmp_path / "class").mkdir()
    flag = tmp_path / "served"
    res = _bash(f"rdna_serve_guarded touch {flag}", _env(tmp_path))
    assert res.returncode != 0
    assert "no AMD GPU temperature sensor found" in res.stderr
    assert not flag.exists()


def test_refuses_a_card_already_at_the_limit(four_cards: Path):
    _card(four_cards, 2, "0000:6f:00.0", {"edge": 40000, "junction": 90000})
    flag = four_cards / "served"
    res = _bash(f"rdna_serve_guarded touch {flag}", _env(four_cards))
    assert res.returncode != 0
    assert "a GPU is already at junction 90°C on 0000:6f:00.0" in res.stderr
    assert not flag.exists()


def test_a_trip_stops_the_server_and_exits_90(four_cards: Path):
    got_term = four_cards / "term"
    server = four_cards / "server.sh"
    server.write_text(
        "#!/usr/bin/env bash\n"
        f"trap 'echo TERM > {got_term}; exit 0' TERM\n"
        "while :; do sleep 0.1; done\n"
    )
    env = _env(four_cards, THERMAL_GUARD_INTERVAL="1", THERMAL_GUARD_GRACE="5")
    proc = subprocess.Popen(
        ["bash", "-c", f'source "{COMMON}"; rdna_serve_guarded bash "{server}"'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        time.sleep(1.5)
        assert proc.poll() is None, proc.communicate()
        _card(four_cards, 1, "0000:69:00.0", {"edge": 40000, "mem": 92000})
        _, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 90, err
    assert "THERMAL ABORT: mem 92°C on 0000:69:00.0 (THERMAL_ABORT_C=90)" in err
    assert got_term.read_text().strip() == "TERM"


def test_a_server_that_ignores_sigterm_is_killed(four_cards: Path):
    server = four_cards / "stubborn.sh"
    server.write_text(
        "#!/usr/bin/env bash\ntrap '' TERM\nwhile :; do sleep 0.1; done\n"
    )
    env = _env(
        four_cards,
        THERMAL_ABORT_C="50",
        THERMAL_GUARD_INTERVAL="1",
        THERMAL_GUARD_GRACE="1",
    )
    proc = subprocess.Popen(
        ["bash", "-c", f'source "{COMMON}"; rdna_serve_guarded bash "{server}"'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        time.sleep(1.5)
        _card(four_cards, 0, "0000:64:00.0", {"edge": 51000})
        _, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 90, err
    assert "SIGKILL to it and its descendants" in err


@pytest.fixture()
def venv(tmp_path: Path) -> Path:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    return tmp_path / "venv"


def _plan(venv: Path, *args: str, **env: str):
    full = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("THERMAL_")
        and k not in ("HIP_VISIBLE_DEVICES", "VENV", "MODEL", "VIRTUAL_ENV")
    }
    full.update(VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", **env)
    return subprocess.run(
        ["bash", str(LAUNCHER), "RECIPE=27b-awq", *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )


def test_print_plan_shows_the_limit(venv: Path):
    res = _plan(venv, "THERMAL_ABORT_C=85")
    assert res.returncode == 0, res.stderr
    assert (
        "# thermal:   stop at 85°C on any AMD GPU sensor (THERMAL_ABORT_C; "
        "exit status 90)" in res.stdout
    )


def test_print_plan_refuses_a_raised_limit(venv: Path):
    res = _plan(venv, "THERMAL_ABORT_C=95")
    assert res.returncode != 0
    assert "from 1 to 90" in res.stdout + res.stderr
