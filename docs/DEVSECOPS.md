# Build and security pipeline

## Workflows

| Workflow | Runs on | What it does |
|---|---|---|
| `ci.yml` | Push to `main`/`dev`/`release/**`, every PR | Lock and model checks, lint, tests on Windows and Linux |
| `security.yml` | Push, PR, weekly | Secret scan (gitleaks, full history), Bandit, pip-audit, actionlint + zizmor, dependency review on PRs |
| `codeql.yml` | Push, PR, weekly | CodeQL security-extended queries for Python |
| `build.yml` | PRs touching code or packaging, push to `dev`, manual | PyInstaller bundles for Windows and Linux plus selfcheck; 14-day artifacts |
| `release.yml` | Tag `v*.*.*`, manual dry run | Builds, SBOM, `SHA256SUMS`, provenance attestation, draft GitHub Release |

Rules all workflows follow: third-party actions pinned to a commit SHA (Dependabot
keeps them current), `permissions: contents: read` by default with write scopes only
on the job that needs them, checkout without stored credentials, no
`pull_request_target`, scanners downloaded with a fixed sha256.

## Dependencies

- `requirements.txt` — the direct dependencies, edited by hand.
- `requirements.lock` — every package with hashes, generated from it. CI fails when
  the two disagree. After changing `requirements.txt`:
  ```bash
  uv pip compile --universal --generate-hashes --python-version 3.12 --no-header \
      requirements.txt -o requirements.lock
  ```
- CI, builds and releases install from the lock with `--require-hashes`.
- `packaging/requirements-build.txt` — hash-locked PyInstaller, from `requirements-build.in`.

## Models

`packaging/model-checksums.sha256` lists every shipped model with its sha256. CI
fails if a tracked model is missing from the list or does not match it. When a model
is added or replaced on purpose:
```bash
python tools/ci/check_models.py --update
```
and commit the list with the model. Only listed models are put into bundles.

## Builds

```bash
python -m pip install --require-hashes -r packaging/requirements-build.txt
python tools/ci/build_bundle.py            # Linux: run under xvfb-run -a
```
Unset `PYTHONPATH` first (the build refuses to run with it, so packages from e.g. a
ROS install cannot enter the bundle). The result is in `release-assets/`. `packaging/flod_mas.spec` keeps the source
layout inside the bundle because the application finds `config.json` and its models
next to its own modules. `flod-mas-selfcheck` in the bundle verifies model checksums,
loads the libraries and opens every ONNX model; it needs no camera or window.

## Local checks before pushing

```bash
pre-commit install            # once; then runs on every commit
pre-commit run --all-files
tools/ci/check_lock.sh
python tools/ci/check_models.py --tracked
uvx bandit==1.9.4 -c pyproject.toml -r . -ll -b .bandit-baseline.json
uvx pip-audit==2.10.1 -r requirements.lock --require-hashes --disable-pip
gitleaks git --redact .
```

Tests read settings from `Documents/data files`; run them with a clean home to match
CI: `HOME=$(mktemp -d) xvfb-run -a python -m unittest discover -s tests -p "test_*.py"`.

## Bandit baseline

`.bandit-baseline.json` records the findings present when scanning started. CI fails
only on new medium or high findings. Fixing an old finding needs no baseline change;
regenerate the baseline only to accept a reviewed finding, in its own pull request.

## Tracking

Findings go to GitHub Issues (labels `security`, `ci`, `packaging`, ...). Findings
that could be exploited before they are fixed go to private Security Advisories,
never to public issues.
