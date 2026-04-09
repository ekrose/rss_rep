#!/bin/bash
# =============================================================================
# Run estimate_variance_robustness.do for each specification in robust_options.txt
# Supports parallel execution: each Stata process runs in an isolated temp
# directory so log files from concurrent processes don't collide.
#
# Usage: ./run_robustness.sh [n_workers] [max_iterations]
#   n_workers:      number of parallel Stata processes (default: 4)
#   max_iterations: run only the first N iterations (default: all)
#
# Examples:
#   ./run_robustness.sh              # 4 workers, all 812 iterations
#   ./run_robustness.sh 8            # 8 workers, all iterations
#   ./run_robustness.sh 4 10         # 4 workers, first 10 only
#   ./run_robustness.sh 1            # sequential (1 worker), all iterations
#
# Output:
#   temp/robust/resids_iter{N}.dta   — teacher-year residuals per specification
#   temp/robust/iter{N}.log          — Stata log per specification
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_ROOT/.envrc"
source "$PROJECT_ROOT/.venv/bin/activate"

DOFILE="code/estimate_variance_robustness.do"
OPTIONS_FILE="code/robust_options.txt"
LOGDIR="temp/robust"

NWORKERS="${1:-4}"
MAX_ITER="${2:-0}"

mkdir -p "$LOGDIR"

echo "Running robustness specifications with $NWORKERS parallel worker(s)"
[ "$MAX_ITER" -gt 0 ] 2>/dev/null && echo "Limiting to first $MAX_ITER iterations"
echo ""

# Directory to collect error markers (one file per failed iteration)
ERROR_DIR=$(mktemp -d)
trap 'rm -rf "$ERROR_DIR"' EXIT

# --------------------------------------------------------------------------
# Worker: runs a single Stata iteration in an isolated temp directory
# --------------------------------------------------------------------------
run_one() {
    local iter=$1
    local covdesign="$2"

    # Skip if output already exists
    if [ -f "$PROJECT_ROOT/$LOGDIR/resids_iter${iter}.dta" ]; then
        echo "[iter $iter] Output exists, skipping"
        return 0
    fi

    # Create isolated working directory with symlinks so Stata's
    # relative paths (code/, temp/) resolve to the real project dirs
    local jobdir
    jobdir=$(mktemp -d)
    ln -s "$PROJECT_ROOT/code" "$jobdir/code"
    ln -s "$PROJECT_ROOT/temp" "$jobdir/temp"

    # Run Stata from the isolated directory
    ( cd "$jobdir" && "$STATA" -b do "$PROJECT_ROOT/$DOFILE" ${iter} ${covdesign} ) 2>/dev/null

    # Wait for Stata to finish (macOS -b may return immediately)
    local logfile="$jobdir/estimate_variance_robustness.log"
    while [ ! -f "$logfile" ] || ! grep -q "end of do-file" "$logfile" 2>/dev/null; do
        sleep 2
    done

    # Check the log for errors
    if [ -f "$logfile" ]; then
        if grep -q "^r(" "$logfile"; then
            echo "[iter $iter] ERROR — see $LOGDIR/iter${iter}.log"
            mv "$logfile" "$PROJECT_ROOT/$LOGDIR/iter${iter}.log"
            touch "$ERROR_DIR/iter${iter}"
            rm -rf "$jobdir"
            return 1
        fi
        mv "$logfile" "$PROJECT_ROOT/$LOGDIR/iter${iter}.log"
    fi

    rm -rf "$jobdir"
    echo "[iter $iter] OK"
    return 0
}

# --------------------------------------------------------------------------
# Wait until the number of running background jobs drops below NWORKERS
# --------------------------------------------------------------------------
wait_for_slot() {
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NWORKERS" ]; do
        sleep 0.5
    done
}

# --------------------------------------------------------------------------
# Pre-process the options file: join continuation lines (starting with ">")
# into a single specification per line
# --------------------------------------------------------------------------
specs=()
iters=()
current_iter=0
current_spec=""

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue

    if [[ "$line" == ">"* ]]; then
        # Continuation line: append (strip leading "> ")
        current_spec="$current_spec ${line:2}"
    else
        # Save the previous spec (if any)
        if [ -n "$current_spec" ]; then
            current_iter=$((current_iter + 1))
            specs+=("$current_spec")
            iters+=("$current_iter")
        fi
        current_spec="$line"
    fi
done < "$OPTIONS_FILE"
# Save the last spec
if [ -n "$current_spec" ]; then
    current_iter=$((current_iter + 1))
    specs+=("$current_spec")
    iters+=("$current_iter")
fi

total=${#specs[@]}
echo "Total specifications: $total"
echo ""

# --------------------------------------------------------------------------
# Dispatch iterations in parallel
# --------------------------------------------------------------------------
launched=0
for i in "${!specs[@]}"; do
    iter="${iters[$i]}"
    covdesign="${specs[$i]}"

    # Respect max iterations limit
    if [ "$MAX_ITER" -gt 0 ] 2>/dev/null && [ "$launched" -ge "$MAX_ITER" ]; then
        break
    fi

    wait_for_slot
    run_one "$iter" "$covdesign" &
    launched=$((launched + 1))
done

# Wait for all remaining background jobs to finish
wait

# --------------------------------------------------------------------------
# Report results
# --------------------------------------------------------------------------
n_errors=$(find "$ERROR_DIR" -type f | wc -l | tr -d ' ')

echo ""
echo "========================================"
echo "Completed $launched of $total iterations."
if [ "$n_errors" -gt 0 ]; then
    echo "WARNING: $n_errors iteration(s) had errors. Check $LOGDIR/iter*.log"
    exit 1
else
    echo "All iterations succeeded."
fi

echo ""
echo "Producing Figure A4..."
python "$PROJECT_ROOT/code/vcov_robustness.py"
