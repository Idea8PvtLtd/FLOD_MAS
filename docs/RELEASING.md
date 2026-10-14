# Releasing

## Versions

Semantic versions `MAJOR.MINOR.PATCH`, tagged as `vMAJOR.MINOR.PATCH`.

| Tag | Made from | Published as | Used for |
|---|---|---|---|
| `vX.Y.Z-rc.N` | `dev` | Pre-release | Site acceptance testing |
| `vX.Y.Z` | `main` only | Release | Production |

The release workflow refuses a stable tag whose commit is not on `main`.

## Cutting a release candidate

1. Make sure CI, Security and Build are green on `dev`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading
   (through a pull request).
3. Tag and push:
   ```bash
   git switch dev && git pull
   git tag -a v1.1.0-rc.1 -m "FLOD MAS 1.1.0 release candidate 1"
   git push origin v1.1.0-rc.1
   ```
4. The **Release** workflow builds Windows and Linux, runs the selfcheck, and
   creates a **draft** pre-release with the archives, `SHA256SUMS`, the SBOM and a
   provenance attestation.
5. Review the draft (assets present, notes correct) and press **Publish**.

## Cutting a stable release

1. Open a pull request `dev` → `main`; merge it after review and green checks.
2. Tag the merge commit on `main` as `vX.Y.Z` and push the tag.
3. Review and publish the draft as above.

## Dry run

Actions → Release → *Run workflow* with a version such as `0.0.0-dryrun.1` builds
everything and keeps it as a workflow artifact for 7 days. Nothing is published.

## Rolling back

- Packaged installs: install the previous release's archive; settings in
  `Documents\data files` are not touched.
- App Starter installs: the starter keeps the last 3 builds and marks a build that
  fails to start as bad. To withdraw a bad commit, revert it on the deployed branch.
- Mark a withdrawn GitHub release as a pre-release and state why in its notes; do
  not delete it, so its checksums stay verifiable.

## Not yet in place

- Windows code signing (unsigned builds show a SmartScreen warning). Tracked as an issue.
- The App Starter installs source from a branch, not these release archives.
  Switching factory delivery to releases is tracked as an issue.
