#!/bin/bash
# Run estimate_variance_robustness.do for each specification in robust_options.txt
# Usage: ./run_robustness.sh [max_iterations]
#   max_iterations: optional, run only the first N iterations (default: all)

set -e

STATA="/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se"
DOFILE="code/estimate_variance_robustness.do"
OPTIONS_FILE="code/robust_options.txt"
LOGDIR="temp/robust"
export PROJECT_DATA_DIR="$HOME/Documents/Data_rss"

# Create output directory
mkdir -p "$LOGDIR"

# Optional: limit number of iterations
MAX_ITER="${1:-0}"

iter=1
while IFS= read -r covdesign || [ -n "$covdesign" ]; do
    # Skip empty lines
    [ -z "$covdesign" ] && continue

    # Stop if we've reached the max
    if [ "$MAX_ITER" -gt 0 ] && [ "$iter" -gt "$MAX_ITER" ]; then
        break
    fi

    echo "=== Iteration $iter ==="
    echo "  covdesign: $covdesign"

    # Stata `0' = full arg string, `1' = first token.
    # The .do file strips the leading number from `0' to get covdesign.
    # Pass as: iter covdesign  (single argument string after the .do file)
    "$STATA" -b do "$DOFILE" ${iter} ${covdesign}

    # Move the log to the robust directory
    LOGFILE="estimate_variance_robustness.log"
    if [ -f "$LOGFILE" ]; then
        # Check for errors
        if grep -q "^r(" "$LOGFILE"; then
            echo "  ERROR in iteration $iter — see $LOGDIR/iter${iter}.log"
            mv "$LOGFILE" "$LOGDIR/iter${iter}.log"
            exit 1
        else
            echo "  OK"
            mv "$LOGFILE" "$LOGDIR/iter${iter}.log"
        fi
    fi

    iter=$((iter + 1))
done < "$OPTIONS_FILE"

echo ""
echo "Completed $((iter - 1)) iterations successfully."
