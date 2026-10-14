#!/usr/bin/env bash
# Install a pinned scanner release binary into ~/.local/bin after checking its sha256.
#   tools/ci/install_scanner.sh gitleaks|actionlint
# To upgrade, change the version and the sha256 from the release's checksums file.
set -euo pipefail
case "$1" in
  gitleaks)
    url=https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
    sum=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb ;;
  actionlint)
    url=https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz
    sum=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8 ;;
  *) echo "unknown scanner: $1" >&2; exit 2 ;;
esac
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl -fsSL --proto '=https' --tlsv1.2 -o "$work/pkg.tar.gz" "$url"
echo "$sum  $work/pkg.tar.gz" | sha256sum -c -
tar -xzf "$work/pkg.tar.gz" -C "$work" "$1"
mkdir -p "$HOME/.local/bin"
install -m 0755 "$work/$1" "$HOME/.local/bin/$1"
echo "$HOME/.local/bin" >> "${GITHUB_PATH:-/dev/null}"
