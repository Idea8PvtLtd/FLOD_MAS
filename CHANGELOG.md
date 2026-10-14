# Changelog

All notable changes to FLOD MAS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): MAJOR for changes that need new
calibration or settings on site, MINOR for new features (off by default),
PATCH for fixes.

## [Unreleased]

### Added
- CI on Windows and Linux, security scanning (secrets, code, dependencies,
  workflows, CodeQL) and Dependabot.
- Packaged Windows and Linux builds with a bundled `flod-mas-selfcheck`.
- Tagged releases with checksums, SBOM and build provenance.
- Hash-locked dependencies (`requirements.lock`) and a model checksum list.
- Licence, third-party notices, security policy and release procedure.
