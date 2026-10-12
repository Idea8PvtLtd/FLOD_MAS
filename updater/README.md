# App Starter (updater)

`updater.py` installs the newest commit of one branch of the repository into
`releases/<commit>` beside itself, installs the Python packages when
`requirements.txt` changed, and starts `main_1366.py`. It checks **once, when it is
started**, and closes itself after the application is running. Without internet, or when
any step fails or takes too long, it starts the build that is already installed.

The starter itself needs `customtkinter` (`updater/requirements.txt`) and Git. It does
not notice an application that is still running from an earlier start; close that one
first, or the second cannot open the camera.

## Settings (optional)

Copy `updater.example.json` to `updater.json` beside the starter (beside the `.exe` when it
is packaged). Without the file nothing changes.

| Key | Default | Meaning |
|---|---|---|
| `branch` | `"main"` | The branch that is installed. Every push to that branch reaches this PC at its next start. Point the inspection PC at a branch that only receives commits whose tests passed (for example `release`) and keep `main` for development. |
| `start_check_seconds` | `0` (off) | After an update the new build is watched for this many seconds. If it exits with an error in that time, the previous build is started again and that commit is not installed again; a newer commit is. `20` is a reasonable value. |

## What the tests cover

`tests/test_updater.py` replaces git, pip, the window and the application with stand-ins
and checks the decisions: failed and stalled installs, a rebuilt environment, failures in
the fallback launch, time limits, the branch setting and the start check. The starter
itself has to be tried on the inspection PC; the automated tests do not open its window.

The repository's tests run on Windows for every push and pull request
(`.github/workflows/tests.yml`). Move the deployed branch only to commits that passed.
