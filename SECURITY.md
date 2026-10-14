# Security policy

FLOD MAS runs on production inspection PCs. Security problems are handled
privately until a fixed version is released.

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub private vulnerability reporting:
[Report a vulnerability](https://github.com/Idea8PvtLtd/FLOD_MAS/security/advisories/new).

Include what is affected (file, version or commit), how it can be triggered and
what an attacker gains. You will get an acknowledgement within 5 working days and
a plan within 15 working days.

## Supported versions

| Version | Security fixes |
|---|---|
| Latest stable release | Yes |
| Previous minor release | Critical fixes only, for 6 months |
| Release candidates, branches | No |

## How releases are protected

- Every change to `dev` and `main` goes through a pull request with passing
  CI and security checks.
- Dependencies are hash-locked; model files are checked against a checksum list.
- Release builds are made only by GitHub Actions, published with `SHA256SUMS`,
  an SBOM and a signed build-provenance attestation. See
  [docs/RELEASING.md](docs/RELEASING.md).
