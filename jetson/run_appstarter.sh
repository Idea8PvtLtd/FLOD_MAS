#!/usr/bin/env bash
# Starts the updater, which fetches the newest build of the branch in
# updater.json, installs its packages when they changed, and launches the app.
# The updater's and the app's output both go to starter.log beside this script.
cd "$(dirname "$0")"
export GIT_TERMINAL_PROMPT=0     # no terminal at login: fail instead of waiting for a password
exec ./starter-venv/bin/python updater.py >> starter.log 2>&1
