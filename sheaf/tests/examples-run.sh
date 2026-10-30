#!/usr/bin/env bash
#
# Run all Sheaf examples listed in examples-manifest.txt.
#
# Usage:
#   sheaf/tests/examples-run.sh
#   SHEAF_DEVICE=metal ./examples-run.sh
#   SHEAF_DEVICE=cuda  ./examples-run.sh
#
# For each example group, the runner creates a fresh temporary working
# directory, copies the required sources and inputs, executes the examples
# in manifest order, runs the associated validators, checks the expected
# artifacts, and reports PASS/FAIL. The script exits with status 1 if any
# example fails.
# set -u

# paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHEAF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SHEAF_DIR/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/examples"
MANIFEST="$SCRIPT_DIR/examples-manifest.txt"
CHECKS_FILE="$SCRIPT_DIR/examples-checks.shf"

# sheaf binary
SHEAF_BIN=""
for candidate in \
    "$SHEAF_DIR/target/release/sheaf" \
    "$SHEAF_DIR/target/debug/sheaf"; do
    if [[ -x "$candidate" ]]; then
        SHEAF_BIN="$candidate"
        break
    fi
done
if [[ -z "$SHEAF_BIN" ]]; then
    SHEAF_BIN="$(command -v sheaf 2>/dev/null || true)"
fi
if [[ -z "$SHEAF_BIN" ]]; then
    echo "examples-run: sheaf binary not found. Run: cd $SHEAF_DIR && cargo build" >&2
    exit 1
fi

DEVICE="${SHEAF_DEVICE:-cpu}"

# preflight
if [[ ! -f "$MANIFEST" ]]; then
    echo "examples-run: manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$CHECKS_FILE" ]]; then
    echo "examples-run: checks file not found: $CHECKS_FILE" >&2
    exit 1
fi

# logging helpers
if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_GREY=$'\033[90m'; C_BOLD=$'\033[1m'; C_NC=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_GREY=""; C_BOLD=""; C_NC=""
fi

log_pass () { printf "  %sPASS%s  %s\n" "$C_GREEN" "$C_NC" "$1"; }
log_fail () { printf "  %sFAIL%s  %s  -- %s\n" "$C_RED" "$C_NC" "$1" "$2"; }
log_info () { printf "  %s%s%s\n" "$C_GREY" "$1" "$C_NC"; }

# validators: bash+awk checks against example output.
#
# Each validator receives the log paths it needs and outputs one of:
#   PASS
#   FAIL: <reason>
# Anything else is treated as FAIL: unexpected.

validate_mlp () {
    local stdout="$1"
    # Extract the 4 prediction values "x.yyy" into 4 shell variables.
    # Format observed: "  [A. B.] -> V" where A,B in {0,1} and V is float.
    local got_00 got_01 got_10 got_11
    got_00="$(grep -E '\[0\. 0\.\] -> ' "$stdout" | tail -n1 | sed -E 's/.* -> //')"
    got_01="$(grep -E '\[0\. 1\.\] -> ' "$stdout" | tail -n1 | sed -E 's/.* -> //')"
    got_10="$(grep -E '\[1\. 0\.\] -> ' "$stdout" | tail -n1 | sed -E 's/.* -> //')"
    got_11="$(grep -E '\[1\. 1\.\] -> ' "$stdout" | tail -n1 | sed -E 's/.* -> //')"

    for v in "$got_00" "$got_01" "$got_10" "$got_11"; do
        if [[ -z "$v" ]]; then
            echo "FAIL: missing prediction line(s)"; return 0
        fi
    done

    # Works with BSD awk
    awk -v p00="$got_00" -v p01="$got_01" -v p10="$got_10" -v p11="$got_11" '
        function check(p, label, lo, hi,    ok) {
            if (p+0 < lo)                    { print "FAIL: " label "=" p " < " lo; exit 1 }
            if (p+0 > hi)                    { print "FAIL: " label "=" p " > " hi; exit 1 }
        }
        BEGIN {
            check(p00, "[0,0]", 0.0,  0.1)
            check(p01, "[0,1]", 0.9,  1.01)
            check(p10, "[1,0]", 0.9,  1.01)
            check(p11, "[1,1]", 0.0,  0.1)
            print "PASS"
        }
    '
}

validate_hydra () {
    local stdout="$1"
    if ! grep -q -- "--- Training Successful ---" "$stdout"; then
        echo "FAIL: missing '--- Training Successful ---'" ; return 0
    fi
    if ! grep -q -- "[Evolution]" "$stdout"; then
        echo "FAIL: missing '[Evolution]' (network did not grow)" ; return 0
    fi
    # Find the last Epoch line's loss
    local last_loss
    last_loss="$(grep -E '^Epoch [0-9]+ \| Loss: ' "$stdout" \
        | tail -n 1 \
        | awk -F'Loss: ' '{print $2}' \
        | awk '{print $1}')"
    if [[ -z "$last_loss" ]]; then
        echo "FAIL: no 'Epoch ... | Loss: ...' line found"; return 0
    fi
    # loss should be <= 0.01
    awk -v l="$last_loss" 'BEGIN { exit (l+0 < 0.01) ? 0 : 1 }' \
        && echo "PASS" \
        || echo "FAIL: final loss $last_loss >= 0.01 (training did not converge)"
}

validate_nanoGPT_sample () {
    local stdout="$1" stderr="$2"
    if ! grep -q -- "Generating 500 tokens..." "$stdout"; then
        echo "FAIL: 'Generating 500 tokens...' missing"; return 0
    fi
    if ! grep -Eq -- 'jit: compiling gpt-forward \[|jit: gpt-forward \(cached\)' "$stderr"; then
        echo "FAIL: gpt-forward did not compile through JIT"; return 0
    fi
    local sep_count
    sep_count="$(grep -c -- '^---$' "$stdout" || true)"
    if [[ $sep_count -lt 2 ]]; then
        echo "FAIL: expected at least 2 '---' delimiter lines, got $sep_count"; return 0
    fi
    local size
    size="$(wc -c <"$stdout" | tr -d ' ')"
    if [[ $size -lt 200 ]]; then
        echo "FAIL: stdout only $size bytes, sampling likely truncated"; return 0
    fi
    echo "PASS"
}

# Run an example
#
# Args from the manifest file:
#   $1 group directive name (may be empty)
#   $2 entry (.shf file)
#   $3 inputs (comma-separated paths from examples/<group>/)
#   $4 artifacts (comma-separated paths under workdir)
#   $5 validator
#   $6 timeout seconds
#
# Sets global: LAST_STATUS ("pass" or "fail: <reason>")
LAST_STATUS=""

record_pass () { LAST_STATUS="pass"; }
record_fail () { LAST_STATUS="fail: $1"; }

run_example () {
    local group="$1" entry="$2" inputs="$3" artifacts="$4" \
          validator="$5" timeout="$6"

    local group_dir="$EXAMPLES_DIR/$group"
    if [[ ! -d "$group_dir" ]]; then
        record_fail "group dir not found: $group_dir"
        return
    fi

    # Resolve or create workdir for this group.
    local workdir
    if [[ -n "${GROUP_WD[$group]:-}" ]]; then
        workdir="${GROUP_WD[$group]}"
    else
        workdir="$ROOT_WORK/$group"
        mkdir -p "$workdir"
        GROUP_WD[$group]="$workdir"

        # Copy all .shf from the group dir.
        # `cp ... -t` is GNU-only, we use a find|while loop for BSD compat.
        while IFS= read -r -d '' f; do
            cp "$f" "$workdir/"
        done < <(find "$group_dir" -maxdepth 1 -type f -name '*.shf' -print0)

        # Copy examples-checks.shf into the workdir so `(use examples-checks)`
        # resolves.
        cp "$CHECKS_FILE" "$workdir/"
        if [[ "$validator" == "clevr" ]]; then
            cp "$SCRIPT_DIR/regressions.shf" "$workdir/" || {
                record_fail "could not copy shared CLEVR regression"
                return
            }
        fi

        # Copy inputs (comma list of relative paths under examples/<group>/).
        if [[ -n "$inputs" && "$inputs" != "none" ]]; then
            IFS=',' read -r -a items <<< "$inputs"
            for item in "${items[@]}"; do
                # trim whitespace
                item="${item## }"; item="${item%% }"
                [[ -z "$item" ]] && continue
                local src="$group_dir/$item"
                if [[ ! -e "$src" ]]; then
                    record_fail "input not found: $src"
                    return
                fi
                # Ensure the parent directory in the workdir exists so we
                # can cp -R with the leaf name.
                local parent_rel
                parent_rel="$(dirname "$item")"
                if [[ "$parent_rel" != "." ]]; then
                    mkdir -p "$workdir/$parent_rel"
                fi
                cp -R "$src" "$workdir/$item"
            done
        fi
    fi

    # Verify the entry script exists in the workdir.
    if [[ ! -f "$workdir/$entry" ]]; then
        record_fail "entry not copied into workdir: $entry"
        return
    fi
    # Run with timeout. Redirect to per-example log files.
    local base_log
    base_log="$ROOT_LOG/${group}_$(basename "$entry" .shf)"
    if [[ "$validator" == "nanoGPT-resume" ]]; then
        base_log="${base_log}_resume"
    fi
    local stdout_log="$base_log.stdout"
    local stderr_log="$base_log.stderr"
    : >"$stdout_log"; : >"$stderr_log"

    if [[ "$validator" == "nanoGPT-train" || "$validator" == "nanoGPT-resume" ]]; then
        local checkpoint="$workdir/out-shakespeare-char/ckpt.safetensors"
        if [[ "$validator" == "nanoGPT-train" && -e "$checkpoint" ]]; then
            record_fail "fresh training already has a checkpoint"
            return
        fi
        if [[ "$validator" == "nanoGPT-resume" && ! -f "$checkpoint" ]]; then
            record_fail "resume checkpoint is missing"
            return
        fi
        rm -rf -- "$workdir/__sheaf__"
    fi

    log_info "→ $group/$entry  (device=$DEVICE, timeout=${timeout}s)"

    local -a sheaf_args=(--device "$DEVICE")
    # NanoGPT model and training functions must compile instead of falling back.
    if [[ "$validator" == "nanoGPT-sample" || "$validator" == "nanoGPT-train" || "$validator" == "nanoGPT-resume" ]]; then
        sheaf_args+=(-v)
    fi

    local -a program_args=("$entry")
    if [[ "$validator" == "clevr" ]]; then
        log_info "  using the shared deterministic CLEVR regression"
        program_args=(-c '(use data) (use model) (use regressions)
            (check-clevr-params (io "load" "params.pkl"))
            (print "CLEVR regression passed")')
    fi

    local exit_code
    ( cd "$workdir" && timeout --foreground "$timeout" \
        "$SHEAF_BIN" "${sheaf_args[@]}" "${program_args[@]}" \
            >"$stdout_log" 2>"$stderr_log" ) ; exit_code=$?

    if [[ $exit_code -eq 124 ]]; then
        record_fail "timeout after ${timeout}s (exited 124)"
        return
    fi
    if [[ $exit_code -ne 0 ]]; then
        record_fail "sheaf exited $exit_code (logs: ${base_log}.{stdout,stderr})"
        return
    fi

    # Verify artifacts column.
    if [[ -n "$artifacts" && "$artifacts" != "none" ]]; then
        IFS=',' read -r -a arma <<< "$artifacts"
        for a in "${arma[@]}"; do
            a="${a## }"; a="${a%% }"
            [[ -z "$a" ]] && continue
            if [[ ! -e "$workdir/$a" ]]; then
                record_fail "artifact missing: $a"
                return
            fi
        done
    fi

    local val_out=""
    local v_exit=0
    case "$validator" in
        mlp)
            val_out="$(validate_mlp "$stdout_log")" ;;
        hydra)
            val_out="$(validate_hydra "$stdout_log")" ;;
        clevr)
            if grep -qx 'CLEVR regression passed' "$stdout_log"; then
                val_out="PASS"
            else
                val_out="FAIL: shared CLEVR regression did not finish"
            fi ;;
        nanoGPT-sample)
            val_out="$(validate_nanoGPT_sample "$stdout_log" "$stderr_log")" ;;
        nanoGPT-train|nanoGPT-resume)
            local marker="No checkpoint; initializing"
            if [[ "$validator" == "nanoGPT-resume" ]]; then marker="Resuming from"; fi
            if ! grep -Fq -- "$marker" "$stdout_log"; then
                record_fail "expected training path was not taken: $validator"
                return
            fi
            if grep -q -- 'jit: train-step skipped' "$stderr_log"; then
                record_fail "train-step fell back to the interpreter"
                return
            fi
            local compilations
            compilations="$(grep -Fc -- 'jit: compiling train-step [' "$stderr_log" || true)"
            if [[ "$compilations" -ne 1 ]]; then
                record_fail "expected one train-step compilation, got $compilations"
                return
            fi
            val_out="$( cd "$workdir" && \
                timeout --foreground 30 \
                    "$SHEAF_BIN" --device "$DEVICE" -c \
                    '(use examples-checks) (print (check-nanoGPT-train "." "out" "err"))' \
                    2>"$base_log.validator.stderr" )"
            v_exit=$?
            printf '%s\n' "$val_out" >"$base_log.validator.stdout"
            ;;
        *)
            record_fail "unknown validator: $validator"
            return
            ;;
    esac

    # Decide PASS/FAIL.
    # - For embedded bash+awk validators, output is "PASS" or "FAIL: ...".
    # - For nanoGPT-train, the inner `sheaf -c` will exit non-zero if
    #   guard :no-nan breaches
    if [[ $v_exit -ne 0 ]]; then
        record_fail "validator failed (exit $v_exit; check $base_log.validator.*)"
        return
    fi

    # Normalize slight newline mismatches and accept "true" or FAIL:...
    local trimmed
    trimmed="$(printf '%s' "$val_out" | tr -d '[:space:]')"

    if [[ "$trimmed" == "true" ]]; then
        record_pass
    elif [[ "$trimmed" == PASS ]]; then
        record_pass
    elif [[ "$val_out" == FAIL:* ]]; then
        record_fail "$val_out"
    else
        # Validator returned garbage; fail safe.
        record_fail "validator returned unexpected: ${val_out:-<empty>}"
    fi
}

# main
ROOT_LOG=""
ROOT_WORK=""
TMP_ROOT="${TMPDIR:-/tmp}"
TMP_ROOT="${TMP_ROOT%/}"

# Invoked by the EXIT trap.
# shellcheck disable=SC2329
cleanup () {
    local status=$?
    if [[ -n "$ROOT_WORK" ]]; then
        rm -rf -- "$ROOT_WORK"
    fi
    if [[ -n "$ROOT_LOG" && $status -eq 0 ]]; then
        rm -rf -- "$ROOT_LOG"
    elif [[ -n "$ROOT_LOG" && -d "$ROOT_LOG" ]]; then
        printf "\n  logs retained: %s\n" "$ROOT_LOG" >&2
    fi
}
trap cleanup EXIT

if ! ROOT_LOG="$(mktemp -d "$TMP_ROOT/sheaf-examples-logs.XXXXXX")"; then
    printf "examples-run: failed to create log directory\n" >&2
    exit 1
fi
if ! ROOT_WORK="$(mktemp -d "$TMP_ROOT/sheaf-examples-work.XXXXXX")"; then
    printf "examples-run: failed to create work directory\n" >&2
    exit 1
fi

declare -A GROUP_WD

printf "%sRun Sheaf examples%s\n" "$C_BOLD" "$C_NC"
printf "  binary:  %s\n" "$SHEAF_BIN"
printf "  device:  %s\n" "$DEVICE"
printf "  manifest:%s%s%s\n" "$C_GREY" "$MANIFEST" "$C_NC"
printf "  logs:    %s (retained on failure)\n" "$ROOT_LOG"
printf "\n"

pass_count=0
fail_count=0
fails=()

# Read manifest line by line.
#
# We pre-process with awk: emit each of the 6 columns on its own newline.
# Then read 6 lines per row with `read -r`, which preserves empty fields
# correctly (the IFS=$'\t' read approach collapses consecutive tabs in a
# pipeline on macOS, shifting fields; emit-then-read with default newlines
# sidesteps that).

manifest_streams() {
    awk -F'\t' '
        $1 ~ /^#/ || $1 == "" { next }
        { printf "%s\n%s\n%s\n%s\n%s\n%s\n", $1, $2, $3, $4, $5, $6 }
    ' "$MANIFEST"
}

while IFS= read -r group; do
    IFS= read -r entry
    IFS= read -r inputs
    IFS= read -r artifacts
    IFS= read -r validator
    IFS= read -r timeout
    run_example "$group" "$entry" "$inputs" "$artifacts" "$validator" "$timeout"
    if [[ "$LAST_STATUS" == pass ]]; then
        log_pass "$group/$entry"
        pass_count=$((pass_count + 1))
    else
        log_fail "$group/$entry" "$LAST_STATUS"
        for failure_log in "$ROOT_LOG/${group}_$(basename "$entry" .shf)".*; do
            [[ -f "$failure_log" ]] || continue
            printf '\n--- %s ---\n' "$(basename "$failure_log")" >&2
            tail -n 200 "$failure_log" >&2
        done
        fail_count=$((fail_count + 1))
        fails+=("$group/$entry: $LAST_STATUS")
    fi
done < <(manifest_streams)

printf "\n%sResults:%s %d passed, %d failed\n" "$C_BOLD" "$C_NC" "$pass_count" "$fail_count"
if [[ $fail_count -gt 0 ]]; then
    printf "\n%sFailures:%s\n" "$C_BOLD" "$C_NC"
    for f in "${fails[@]}"; do
        printf "  - %s\n" "$f"
    done
    exit 1
fi
exit 0
