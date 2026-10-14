# FLOD MAS

Camera-based inspection of adhesive (glue) strips for production lines. The
operator places the fabric, the application captures it, measures each glue line
with local ONNX models and grades it against the configured width limits.

Proprietary software of Idea8 (Private) Limited. See [LICENSE](LICENSE).

## Supported platforms

| Platform | Status |
|---|---|
| Windows 10/11 x64 | Production (inspection PCs) |
| Linux x86_64 (Ubuntu 24.04) | Builds and tests in CI; camera defaults are tuned for Windows |

Python 3.12 is required when running from source. Packaged builds include their own Python.

## Download

Builds are published on the [Releases](https://github.com/Idea8PvtLtd/FLOD_MAS/releases) page:

| File | Contents |
|---|---|
| `FLOD_MAS-<version>-windows-x64.zip` | Windows application folder |
| `FLOD_MAS-<version>-linux-x86_64.tar.gz` | Linux application folder |
| `FLOD_MAS-<version>-sbom.cdx.json` | Software bill of materials (CycloneDX) |
| `SHA256SUMS` | Checksums of every file in the release |

Versions ending in `-rc.N` are release candidates for site testing, not production.

### Verify a download

Always check a build before installing it on an inspection PC.

```bash
# Linux
sha256sum --check --ignore-missing SHA256SUMS
# Any platform, with the GitHub CLI: proves the file was built by this repository's workflow
gh attestation verify FLOD_MAS-<version>-windows-x64.zip --repo Idea8PvtLtd/FLOD_MAS
```

```powershell
# Windows: compare with the line for the zip in SHA256SUMS
Get-FileHash .\FLOD_MAS-<version>-windows-x64.zip -Algorithm SHA256
```

### Install a packaged build

1. Extract the archive to a folder the operator account can read but not modify
   (for example `C:\Program Files\FLOD_MAS\<version>`).
2. Run `flod-mas-selfcheck` from that folder. It must end with `Bundle is complete.`
3. Start `FLOD_MAS` (`FLOD_MAS.exe` on Windows).

Settings, calibration and captures live in `Documents\data files`, not in the
application folder, so a new version keeps them.

### Run from source (development)

```bash
python -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock   # Windows: .venv\Scripts\python
.venv/bin/python main_1366.py
```

Inspection PCs that use the App Starter (`updater/`) keep updating from source as
before; see [updater/README.md](updater/README.md).

## Documentation

See [docs/README.md](docs/README.md) for the full index: operation, configuration,
calibration, releasing and the build/security pipeline.

## Security

Do not report vulnerabilities in public issues. See [SECURITY.md](SECURITY.md).
