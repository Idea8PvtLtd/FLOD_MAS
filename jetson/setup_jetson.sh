#!/usr/bin/env bash
# One-time setup of an original Jetson Nano 4GB (JetPack 4.6.x, L4T R32.7) for
# FLOD_MAS. Run it as the operator account (not with sudo; it asks when needed):
#
#   bash jetson/setup_jetson.sh
#
# It installs build tools, builds Python 3.8 with tkinter (Ubuntu 18.04 has no
# python3.8-tk), adds swap, sets full power, and creates ~/AppStarter, which
# clones, installs and starts the app at every login. Running it again is safe.
set -euo pipefail

PYTHON_VERSION=3.8.20                     # last Python 3.8 release
APP_HOME="$HOME/AppStarter"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"

step() { printf '\n=== %s\n' "$*"; }

if ! grep -q 'R32' /etc/nv_tegra_release 2>/dev/null; then
    echo "Warning: this is not JetPack 4.6 (L4T R32). The package set targets the original Jetson Nano."
fi

step "System packages"
sudo apt-get update
sudo apt-get install -y build-essential git curl v4l-utils \
    tk-dev libssl-dev libffi-dev zlib1g-dev libbz2-dev liblzma-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev libgdbm-dev uuid-dev
# Camera and serial access without root.
sudo usermod -aG video,dialout "$USER"

step "Python $PYTHON_VERSION with tkinter"
if command -v python3.8 >/dev/null && python3.8 -c 'import tkinter, ssl, venv, sqlite3' 2>/dev/null; then
    echo "python3.8 with tkinter is already installed: $(command -v python3.8)"
else
    build="$(mktemp -d)"
    (
        cd "$build"
        curl -fsSLO "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
        tar -xzf "Python-$PYTHON_VERSION.tgz"
        cd "Python-$PYTHON_VERSION"
        # No --enable-optimizations: on the Nano it doubles a ~40 minute build, and
        # the heavy work runs in numpy / ONNX Runtime / TensorFlow, not in Python.
        ./configure --with-ensurepip=install
        make -j4
        sudo make altinstall          # /usr/local/bin/python3.8; system python3 untouched
    )
    rm -rf "$build"
    python3.8 -c 'import tkinter, ssl, venv' || { echo "Python 3.8 was built without tkinter/ssl"; exit 1; }
fi

step "Swap (4 GB file, on top of JetPack's zram)"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
swapon --show

step "Full power (MAXN) and fixed clocks at every boot"
sudo nvpmodel -m 0
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'EOF'
[Unit]
Description=Run jetson_clocks for steady inspection timing
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-clocks.service

step "AppStarter in $APP_HOME"
mkdir -p "$APP_HOME"
cp "$REPO_ROOT/updater/updater.py" "$APP_HOME/updater.py"
# Keep an existing updater.json, so a branch chosen on this device survives a re-run.
[ -f "$APP_HOME/updater.json" ] || cp "$HERE/updater.json" "$APP_HOME/updater.json"
cp "$HERE/run_appstarter.sh" "$APP_HOME/run_appstarter.sh"
chmod +x "$APP_HOME/run_appstarter.sh"
if [ ! -x "$APP_HOME/starter-venv/bin/python" ]; then
    python3.8 -m venv "$APP_HOME/starter-venv"
fi
"$APP_HOME/starter-venv/bin/python" -m pip install --upgrade pip
"$APP_HOME/starter-venv/bin/python" -m pip install customtkinter==6.0.0

step "Start at login"
mkdir -p "$HOME/.config/autostart"
sed "s|@APP_HOME@|$APP_HOME|g" "$HERE/appstarter.desktop" > "$HOME/.config/autostart/appstarter.desktop"

step "GitHub access"
# The starter runs without a terminal, so it can never answer a password prompt.
# Store the credentials once here; git then reuses them.
git config --global credential.helper store
if git ls-remote https://github.com/Idea8PvtLtd/FLOD_MAS.git HEAD >/dev/null; then
    echo "Repository access OK."
else
    echo "Could not reach the repository. Run this again once access works:"
    echo "  git ls-remote https://github.com/Idea8PvtLtd/FLOD_MAS.git HEAD"
fi

step "Done"
echo "Log out and back in (for the video group), or reboot. AppStarter then starts by itself."
echo "To start it now: $APP_HOME/run_appstarter.sh    (log: $APP_HOME/starter.log)"
