#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared launcher harness for the gfx1030 serve scripts.
#
# Source it, set your knobs, then call the helpers. Nothing here bakes in a
# host-specific path: the tree is derived from this file's location, the venv
# comes from $VENV (or an activated $VIRTUAL_ENV), and the checkpoint comes
# from $MODEL.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/rdna_launcher_common.sh"
#   MODEL=/path/to/ckpt VENV=/path/to/venv
#   rdna_init                 # tree, venv, ROCm SDK, shared vLLM env, caches
#   rdna_tunableop            # honours TUNABLEOP=0/1 and TUNABLEOP_PROFILE
#   rdna_select_attention fa  # fa | triton  (sets $attention_backend)
#   rdna_mtp_args 2           # sets $spec_args and $capture_sizes
#   rdna_kill_stale           # scoped to $VLLM_CACHE_ROOT
#   rdna_serve_guarded "${cmd[@]}"   # in place of exec: the thermal abort
#
# Env knobs (all optional unless noted):
#   VLLM_TREE   repo root        (default: two levels above tools/rdna/)
#   VENV        python venv      (default: active $VIRTUAL_ENV; else error)
#   MODEL       checkpoint path  (launchers that serve a model: required)
#   HIP_VISIBLE_DEVICES / GPUIDS (default: 0,1,2,3)
#   ROCM_SDK_ROOT                (default: /opt/rocm/core-7.14)
#   TUNABLEOP=0|1                (default: 1)
#   TUNABLEOP_PROFILE=<name>     explicit TunableOp profile (validated)
#   VLLM_CACHE_ROOT / TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR /
#   TORCH_EXTENSIONS_DIR         (default: <tree>/cache/*)
#   RUN_CWD                      (default: <tree>/cache/run; never /tmp)
#   THERMAL_ABORT_C              serving stops at this GPU temperature, °C
#                                (default and maximum 90; see rdna_serve_guarded)
#   THERMAL_GUARD_GPUS           PCI addresses the thermal abort watches
#                                (default: every AMD GPU)

_RDNA_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

rdna_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

rdna_require() {
    local name=$1 hint=${2:-}
    if [[ -z ${!name:-} ]]; then
        printf 'ERROR: %s is not set.%s\n' "$name" "${hint:+ $hint}" >&2
        exit 1
    fi
}

rdna_require_model() {
    rdna_require MODEL "Pass MODEL=/path/to/checkpoint (the launcher has no baked-in default)."
}

rdna_init() {
    source_dir=${VLLM_TREE:-$(cd "$_RDNA_COMMON_DIR/../.." && pwd)}
    export source_dir VLLM_TREE=$source_dir

    runtime=${VENV:-${VIRTUAL_ENV:-}}
    # Literal $VIRTUAL_ENV in the message.
    # shellcheck disable=SC2016
    [[ -n $runtime ]] || rdna_die 'set VENV=/path/to/python-venv (or activate one so $VIRTUAL_ENV is set).'
    # RDNA_DRY_RUN lets serve_rdna.sh PRINT the plan for a fork user before the
    # venv exists; it never skips a check on a real launch.
    if [[ ${RDNA_DRY_RUN:-0} != 1 ]]; then
        [[ -x $runtime/bin/python ]] || rdna_die "VENV=$runtime has no bin/python."
    fi
    VENV=$runtime
    export VENV

    ROCM_SDK_ROOT=${ROCM_SDK_ROOT:-/opt/rocm/core-7.14}
    ROCM_SDK_LIB="$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
    ROCM_SDK="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib"
    export ROCM_SDK_ROOT ROCM_SDK_LIB ROCM_SDK
    export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$VENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROCM_HOME=${ROCM_HOME:-$ROCM_SDK_ROOT} ROCM_PATH=${ROCM_PATH:-$ROCM_SDK_ROOT}
    export HIP_PATH=${HIP_PATH:-$ROCM_SDK_ROOT} HIP_ROOT_DIR=${HIP_ROOT_DIR:-$ROCM_SDK_ROOT}
    [[ -d $ROCM_SDK_ROOT/include ]] && export CPATH="$ROCM_SDK_ROOT/include${CPATH:+:$CPATH}"
    [[ -d $ROCM_SDK_ROOT/lib ]] && export LIBRARY_PATH="$ROCM_SDK_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    [[ -d $ROCM_SDK_ROOT/bin ]] && export PATH="$ROCM_SDK_ROOT/bin:$PATH"

    export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${GPUIDS:-0,1,2,3}}
    export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
    # Serving never attaches a GPU profiler: keep the HIP/HSA runtimes from
    # registering with rocprofiler (no tool init, no queue interception).
    # Profiling runs set ROCPROFILER_REGISTER_ENABLED=1.
    export ROCPROFILER_REGISTER_ENABLED=${ROCPROFILER_REGISTER_ENABLED:-0}
    export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
    export VLLM_ROCM_USE_AITER_MOE=${VLLM_ROCM_USE_AITER_MOE:-0}
    export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-0}
    export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}

    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$source_dir/cache/vllm}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$source_dir/cache/triton}
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$source_dir/cache/inductor}
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$source_dir/cache/extensions}
}

rdna_tunableop() {
    # shellcheck source=tools/rdna2_028/tunableop_env.sh
    source "$source_dir/tools/rdna2_028/tunableop_env.sh"
    if [[ ${TUNABLEOP:-1} == 1 ]]; then
        configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$source_dir/tunableop" || return 1
    else
        export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
        export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    fi
}

rdna_select_attention() {
    local mode=${1:-fa}
    if [[ $mode == fa ]]; then
        export VLLM_USE_RDNA2_FA=1
        attention_backend=RDNA_ATTN
        export VLLM_FA_RDNA2_GQA_DECODE=${VLLM_FA_RDNA2_GQA_DECODE:-1}
        unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
    else
        export VLLM_USE_RDNA2_FA=0
        attention_backend=TRITON_ATTN
        export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    fi
    export attention_backend
}

# Sets spec_args and capture_sizes for the sourcing launcher.
# shellcheck disable=SC2034
rdna_mtp_args() {
    local mtp=${1:-0} decode_width sizes mult
    if [[ $mtp == 0 ]]; then
        spec_args=()
        capture_sizes=${CG_SIZES:-'[1,2,4,8]'}
    else
        spec_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp,\"use_local_argmax_reduction\":true}")
        decode_width=$((mtp + 1))
        sizes=""
        for mult in 1 2 4 8; do
            sizes="$sizes$((decode_width * mult)),"
        done
        capture_sizes=${CG_SIZES:-"[${sizes%,}]"}
    fi
}

rdna_run_cwd() {
    local dir=${RUN_CWD:-$source_dir/cache/run}
    mkdir -p "$dir"
    cd "$dir"
}

rdna_kill_stale() {
    local _sig _p _sent
    for _sig in TERM KILL; do
        _sent=0
        for _p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
            [ -r "/proc/$_p/environ" ] || continue
            if tr '\0' '\n' < "/proc/$_p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}$"; then
                if kill -"$_sig" "$_p" 2>/dev/null; then
                    _sent=1
                fi
            fi
        done
        if [[ $_sent == 1 ]]; then
            if [[ $_sig == TERM ]]; then
                sleep 8
            else
                sleep 3
            fi
        fi
    done
}

# --- thermal abort -------------------------------------------------------------
# Every serve launcher stops serving when any watched GPU temperature sensor
# (amdgpu hwmon: edge, junction, memory) reads THERMAL_ABORT_C or more. The
# limit is 90 °C, and a launch may lower it, never raise it. The V620 reports
# junction crit 100 °C and memory crit 98 °C, so 90 °C sits 10 and 8 below.
#
#   rdna_serve_guarded "${cmd[@]}"
#
# goes where a launcher would `exec "${cmd[@]}"`. It refuses to start when no
# sensor can be read or a card is already at the limit. Otherwise it runs the
# server as a child and reads the sensors every THERMAL_GUARD_INTERVAL seconds
# (2). At the first reading at or above the limit it prints "THERMAL ABORT: ..."
# to stderr, sends the server SIGTERM, and after THERMAL_GUARD_GRACE seconds
# (30) SIGKILLs the server and its descendants if it is still running. It then
# exits with status RDNA_THERMAL_EXIT (90), so a service manager can tell a
# thermal abort from a crash (systemd: RestartPreventExitStatus=90). Otherwise
# it exits with the server's status. SIGTERM, SIGINT and SIGHUP to the launcher
# reach the server as SIGTERM, so systemd, podman stop (the launcher is the
# container's PID 1) and Ctrl-C shut it down as before.
#
# Every AMD GPU is watched, not only HIP_VISIBLE_DEVICES: a wrong guess at the
# HIP-to-PCI order would leave the serving cards unwatched. THERMAL_GUARD_GPUS
# (PCI addresses, e.g. 0000:6f:00.0,0000:72:00.0) narrows it, for a host whose
# other cards belong to another job.

RDNA_THERMAL_EXIT=90
RDNA_HWMON_ROOT=${RDNA_HWMON_ROOT:-/sys/class/hwmon}  # override only for testing

# One line per watched sensor: "<pci address> <label> <°C>".
rdna_gpu_temps() {
    local h bdf f v want=",${THERMAL_GUARD_GPUS:-},"
    want=${want// /}
    for h in "$RDNA_HWMON_ROOT"/hwmon*; do
        [[ $(cat "$h/name" 2>/dev/null) == amdgpu ]] || continue
        bdf=$(basename -- "$(readlink -f -- "$h/device")")
        if [[ $want != ,, && $want != *",$bdf,"* && $want != *",${bdf#*:},"* ]]; then
            continue
        fi
        for f in "$h"/temp*_input; do
            v=$(cat "$f" 2>/dev/null) || continue
            [[ $v =~ ^[0-9]+$ ]] || continue
            printf '%s %s %s\n' "$bdf" "$(cat "${f%_input}_label" 2>/dev/null || basename -- "${f%_input}")" "$((v / 1000))"
        done
    done
}

# rdna_gpu_hot LIMIT: the hottest watched sensor at or above LIMIT, e.g.
# "junction 91°C on 0000:6f:00.0" (status 0); status 1 when all read below.
rdna_gpu_hot() {
    rdna_gpu_temps | awk -v lim="$1" '$3 >= lim && (hot == "" || $3 > max) { max = $3; hot = $2 " " $3 "°C on " $1 }
        END { if (hot == "") exit 1; print hot }'
}

# The validated limit on stdout; exits the launcher on a bad THERMAL_ABORT_C.
rdna_thermal_limit() {
    local limit=${THERMAL_ABORT_C:-90}
    if [[ ! $limit =~ ^[0-9]+$ ]] || ((limit < 1 || limit > 90)); then
        rdna_die "THERMAL_ABORT_C must be a whole number of °C from 1 to 90 (got '$limit'): serving stops at 90 °C at the latest."
    fi
    printf '%s\n' "$limit"
}

rdna_descendants() {
    local c
    for c in $(pgrep -P "$1" 2>/dev/null || true); do
        printf '%s\n' "$c"
        rdna_descendants "$c"
    done
}

rdna_serve_guarded() {
    local limit every=${THERMAL_GUARD_INTERVAL:-2} grace=${THERMAL_GUARD_GRACE:-30}
    local pid hot n rc=0 t
    limit=$(rdna_thermal_limit) || exit 1
    [[ $every =~ ^[1-9][0-9]*$ ]] || rdna_die "THERMAL_GUARD_INTERVAL must be a whole number of seconds (got '$every')."
    [[ $grace =~ ^[0-9]+$ ]] || rdna_die "THERMAL_GUARD_GRACE must be a whole number of seconds (got '$grace')."
    n=$(rdna_gpu_temps | wc -l)
    ((n > 0)) || rdna_die "no AMD GPU temperature sensor found (amdgpu hwmon under $RDNA_HWMON_ROOT${THERMAL_GUARD_GPUS:+, THERMAL_GUARD_GPUS=$THERMAL_GUARD_GPUS}); refusing to serve without the thermal abort."
    if hot=$(rdna_gpu_hot "$limit"); then
        rdna_die "a GPU is already at $hot (THERMAL_ABORT_C=$limit); let it cool before serving."
    fi
    printf 'thermal: serving stops at %s°C on any of %s GPU sensors, read every %ss (hottest now: %s)\n' \
        "$limit" "$n" "$every" "$(rdna_gpu_hot 0 || echo '?')" >&2

    "$@" &
    pid=$!
    trap 'kill -TERM "$pid" 2>/dev/null || true' TERM INT HUP
    while kill -0 "$pid" 2>/dev/null; do
        if hot=$(rdna_gpu_hot "$limit"); then
            trap - TERM INT HUP
            printf 'THERMAL ABORT: %s (THERMAL_ABORT_C=%s); stopping the server (pid %s)\n' "$hot" "$limit" "$pid" >&2
            kill -TERM "$pid" 2>/dev/null || true
            for ((t = 0; t < grace; t++)); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                printf 'THERMAL ABORT: the server is still running after %ss; SIGKILL to it and its descendants\n' "$grace" >&2
                # shellcheck disable=SC2046  # one PID per word
                kill -KILL $(rdna_descendants "$pid") "$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
            exit "$RDNA_THERMAL_EXIT"
        fi
        # In the background, so a signal interrupts the wait and reaches the trap at once.
        sleep "$every" &
        wait $! || true
    done
    wait "$pid" || rc=$?
    exit "$rc"
}
