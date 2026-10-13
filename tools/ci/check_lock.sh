#!/usr/bin/env bash
# Fail when requirements.lock no longer matches requirements.txt.
# Versions already in the lock are kept, so this only reports a difference when
# requirements.txt itself changed. To refresh the lock:
#   uv pip compile --universal --generate-hashes --python-version 3.12 --no-header \
#       requirements.txt -o requirements.lock
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
cp requirements.lock "$tmp"
uv pip compile --quiet --universal --generate-hashes --python-version 3.12 --no-header \
    requirements.txt -o "$tmp"
if ! diff -q requirements.lock "$tmp" >/dev/null; then
    echo "requirements.lock is out of date with requirements.txt. Regenerate it:"
    diff -u requirements.lock "$tmp" | grep -E '^[+-][a-zA-Z0-9]' || true
    exit 1
fi
echo "requirements.lock matches requirements.txt."
