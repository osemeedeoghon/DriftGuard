#!/usr/bin/env bash
# Launch the DriftGuard dashboard.
#
# Exists because `streamlit` is only on PATH when the right environment is
# active, and a fresh shell usually has the system Python instead. This picks
# an interpreter that actually has the dependencies, so it works regardless of
# shell state.
set -euo pipefail

cd "$(dirname "$0")"

# Candidate interpreters, most preferred first. A project-local venv wins if
# one exists; otherwise fall back to the Anaconda install, then whatever
# `python3` resolves to.
CANDIDATES=(
    "./.venv/bin/python"
    "/opt/homebrew/anaconda3/bin/python"
    "$(command -v python3 || true)"
)

PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ] \
       && "$candidate" -c "import streamlit" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "error: no Python with Streamlit installed was found." >&2
    echo >&2
    echo "Tried:" >&2
    for candidate in "${CANDIDATES[@]}"; do
        [ -n "$candidate" ] && echo "  $candidate" >&2
    done
    echo >&2
    echo "Create an environment and install the dependencies:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  ./.venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "DriftGuard · using $PYTHON"
exec "$PYTHON" -m streamlit run app.py "$@"
