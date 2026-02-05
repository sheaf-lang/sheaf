#!/bin/sh
# Runs Hydra and reports XLA recompilation stats.
#
# JAX_LOG_COMPILES=1 makes JAX emit a "Compiling ..." line to stderr
# for every XLA kernel compilation. We split the count at the [Evolution]
# event to prove that growing the network does not trigger any new
# compilation.
#
# Usage:  sh examples/hydra/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Capture stdout and stderr into separate temp files
STDOUT=$(mktemp)
STDERR=$(mktemp)
trap "rm -f $STDOUT $STDERR" EXIT

JAX_LOG_COMPILES=1 python -m sheaf "$SCRIPT_DIR/run.shf" >"$STDOUT" 2>"$STDERR"

# Print the training output
cat "$STDOUT"

# Recompilation report
TOTAL=$(grep -c "Compiling" "$STDERR" || true)

# Find the line number in stdout where [Evolution] appears
GROW_LINE=$(grep -n "\[Evolution\]" "$STDOUT" | head -1 | cut -d: -f1)

if [ -z "$GROW_LINE" ]; then
    # No grow happened — all compilations are "before"
    BEFORE=$TOTAL
    AFTER=0
else
    # Find the timestamp of the [Evolution] print in stdout.
    # JAX log lines and stdout are interleaved in real time, so we use
    # the stderr timestamps: find the first "Compiling" line whose
    # timestamp is >= the grow event.  Simpler approach: replay the
    # merged stream and count.
    MERGED=$(mktemp)
    trap "rm -f $STDOUT $STDERR $MERGED" EXIT
    JAX_LOG_COMPILES=1 python -m sheaf "$SCRIPT_DIR/run.shf" >"$MERGED" 2>&1

    BEFORE=$(awk '/\[Evolution\]/{exit} /Compiling/{n++} END{print n+0}' "$MERGED")
    AFTER=$(awk 'found && /Compiling/{n++} /\[Evolution\]/{found=1} END{print n+0}' "$MERGED")
    TOTAL=$((BEFORE + AFTER))
    rm -f "$MERGED"
fi

echo " XLA Recompilation Report"
echo "  Total compilations : $TOTAL"
echo "  Before grow        : $BEFORE"
echo "  After grow         : $AFTER"

if [ "$AFTER" -eq 0 ]; then
    echo "--> 'grow' required zero recompilation"
else
    echo "--> WARNING: 'grow' required $AFTER recompilation(s)."
fi
