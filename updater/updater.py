#!/usr/bin/env python3
"""
App Starter — Vision Inspection updater.

A modern CustomTkinter-based updater that:
  - Checks internet first.
  - If offline   -> launches the existing release immediately.
  - If online    -> fetches the remote; if a new commit exists, downloads it,
                    installs requirements if they changed, prunes old releases,
                    and launches the app.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import customtkinter as ctk

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
APP_TITLE    = "APP STARTER"
APP_SUBTITLE = "Vision Inspection"
APP_VERSION  = "v1.0.0"

REPO_URL = "https://github.com/mmocbm/FLOD_MAS.git"   # never logged
BRANCH = "main"               # default; "branch" in updater.json replaces it
ENTRYPOINT = "main_1366.py"
REQUIREMENTS_FILE = "requirements.txt"

# Python version required by the target app (TensorFlow 2.21 needs <= 3.13).
TARGET_PYTHON_VERSION = "3.12"

# --- detect frozen exe and place working dirs next to the .exe ------------- #
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent

MIRROR_DIR   = ROOT / "repo.git"
RELEASES_DIR = ROOT / "releases"
VENV_DIR     = ROOT / "venv"
STATE_FILE   = ROOT / "state.json"

SETTINGS_FILE = ROOT / "updater.json"

CHECK_INTERVAL   = 60
KEEP_RELEASES    = 3
INTERNET_HOST    = ("github.com", 443)
INTERNET_TIMEOUT = 5

# No external command may wait for ever: a stalled network or a credential prompt
# would otherwise keep the starter open and the application closed. When a limit
# is reached the update is abandoned and the installed build is started.
GIT_CLONE_TIMEOUT = 1800     # first run: the whole repository, with its models
GIT_FETCH_TIMEOUT = 300
GIT_LOCAL_TIMEOUT = 60       # commands that only read the local mirror
EXTRACT_TIMEOUT   = 600
VENV_TIMEOUT      = 600
PIP_TIMEOUT       = 3600     # TensorFlow and MediaPipe on a slow line

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG           = "#0B0F19"      # deep navy background
CARD         = "#151A28"      # card surface
BORDER       = "#232A3D"      # subtle stroke
ACCENT       = "#6366F1"      # indigo
ACCENT_LIGHT = "#818CF8"      # indigo (hover)
SUCCESS      = "#22C55E"
WARNING      = "#F59E0B"
ERROR        = "#EF4444"
TEXT         = "#F1F5F9"
MUTED        = "#94A3B8"
LOG_BG       = "#0A0E17"
LOG_TEXT     = "#CBD5E1"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("app-starter")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def sha256_file(path: Path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def internet_available() -> bool:
    try:
        with socket.create_connection(INTERNET_HOST, timeout=INTERNET_TIMEOUT):
            return True
    except OSError:
        return False


def _git_exe() -> str:
    """Locate git.exe reliably (bundled portable git -> common installs -> PATH)."""
    candidates = [
        ROOT / "git" / "cmd" / "git.exe",
        ROOT / "git" / "bin" / "git.exe",
        Path(r"C:\Program Files\Git\cmd\git.exe"),
        Path(r"C:\Program Files (x86)\Git\cmd\git.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "cmd" / "git.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe",
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except OSError:
            continue
    return "git"


def _find_target_python() -> str | None:
    """Find a Python interpreter matching TARGET_PYTHON_VERSION."""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["py", f"-{TARGET_PYTHON_VERSION}", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True, check=True, timeout=30,
            )
            exe = out.stdout.strip()
            if exe and Path(exe).exists():
                return exe
        except (OSError, subprocess.SubprocessError):
            pass

    for name in (f"python{TARGET_PYTHON_VERSION}", "python3"):
        exe = shutil.which(name)
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe, "-c",
                 "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
                capture_output=True, text=True, check=True, timeout=30,
            )
            if out.stdout.strip() == TARGET_PYTHON_VERSION:
                return exe
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _venv_python_version(venv: Path) -> str | None:
    py = venv_python(venv)
    if not py.exists():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c",
             "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def run_cmd(cmd, check=True, capture=False, timeout=None, **kw):
    """Run a command. ``timeout`` (seconds) raises subprocess.TimeoutExpired."""
    cmd = [str(c) for c in cmd]
    return subprocess.run(cmd, check=check, text=True, capture_output=capture,
                          timeout=timeout, **kw)


def git_env() -> dict:
    """Git must fail rather than ask: nobody is there to answer a prompt."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1000")   # bytes per second ...
    env.setdefault("GIT_HTTP_LOW_SPEED_TIME", "60")      # ... for this long: give up
    return env


def load_settings(path: Path | None = None) -> dict:
    """Optional per-machine settings beside the starter (updater.json).

    branch               which branch is installed. Default "main". Pointing the
                         inspection PC at a branch that only receives tested
                         commits (for example "release") keeps untested pushes
                         to main away from it.
    start_check_seconds  after an update, how long the new build is watched.
                         If it exits with an error inside that time the previous
                         build is started again and the bad commit is skipped
                         until a newer one arrives. Default 0: no check.
    A missing or unreadable file, or a bad value, gives the defaults.
    """
    data = load_json(path or SETTINGS_FILE, {})
    if not isinstance(data, dict):
        data = {}
    branch = data.get("branch", BRANCH)
    if not isinstance(branch, str) or not branch.strip():
        branch = BRANCH
    seconds = data.get("start_check_seconds", 0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 <= seconds <= 600:
        seconds = 0
    return {"branch": branch.strip(), "start_check_seconds": float(seconds)}


# --------------------------------------------------------------------------- #
# Git wrapper — never logs the URL
# --------------------------------------------------------------------------- #
class GitRepo:
    def __init__(self, url: str, branch: str, mirror: Path):
        self.url = url
        self.branch = branch
        self.mirror = mirror

    def _git(self, *args, timeout=GIT_LOCAL_TIMEOUT, **kw):
        return run_cmd([_git_exe(), f"--git-dir={self.mirror}", *args],
                       timeout=timeout, env=git_env(), **kw)

    def ensure_cloned(self, log):
        if self.mirror.exists():
            return
        log("Connecting to remote repository …")
        try:
            run_cmd([_git_exe(), "clone", "--mirror", self.url, str(self.mirror)],
                    timeout=GIT_CLONE_TIMEOUT, env=git_env())
        except (subprocess.SubprocessError, OSError):
            # A half-made mirror would be taken for a complete one next time.
            shutil.rmtree(self.mirror, ignore_errors=True)
            raise

    def fetch(self, log):
        log("Checking for updates …")
        self._git("fetch", "--prune", "origin", timeout=GIT_FETCH_TIMEOUT)

    def rev(self) -> str:
        return self._git("rev-parse", f"refs/heads/{self.branch}",
                         capture=True).stdout.strip()

    def extract(self, sha: str, dest: Path, log):
        dest.mkdir(parents=True, exist_ok=True)
        log(f"Downloading build {sha[:8]} …")
        proc = subprocess.Popen(
            [_git_exe(), f"--git-dir={self.mirror}", "archive", "--format=tar", sha],
            stdout=subprocess.PIPE, env=git_env(),
        )
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                try:
                    tf.extractall(dest, filter="data")
                except TypeError:
                    tf.extractall(dest)
        finally:
            proc.stdout.close()
        try:
            code = proc.wait(timeout=EXTRACT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        if code != 0:
            raise RuntimeError(f"Failed to extract build {sha}")


# --------------------------------------------------------------------------- #
# Dependency management
# --------------------------------------------------------------------------- #
def ensure_venv(venv: Path, log) -> bool:
    """Make sure the environment exists. True when it was (re)built, and so is empty."""
    existing = _venv_python_version(venv)
    if existing == TARGET_PYTHON_VERSION:
        return False

    if existing is not None:
        log(f"Rebuilding environment (Python {existing} → {TARGET_PYTHON_VERSION}) …")
        shutil.rmtree(venv, ignore_errors=True)

    target = _find_target_python()
    if not target:
        raise RuntimeError(
            f"Python {TARGET_PYTHON_VERSION} was not found.\n"
            f"Install it from https://www.python.org/downloads/ and enable the 'py' launcher."
        )

    log(f"Preparing runtime environment (Python {TARGET_PYTHON_VERSION}) …")
    run_cmd([target, "-m", "venv", str(venv)], timeout=VENV_TIMEOUT)
    run_cmd([venv_python(venv), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
            timeout=PIP_TIMEOUT)
    return True


def install_requirements(venv: Path, req: Path, log):
    log("Installing required components …")
    run_cmd([venv_python(venv), "-m", "pip", "install", "--no-input", "-r", str(req)],
            timeout=PIP_TIMEOUT)


def prune_releases(releases: Path, keep: int, current_sha: str, log):
    dirs = [d for d in releases.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    protected = {current_sha}
    for d in dirs:
        if len(protected) >= keep:
            break
        protected.add(d.name)
    for d in dirs:
        if d.name not in protected:
            log(f"Cleaning up old build {d.name[:8]} …")
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Child process manager
# --------------------------------------------------------------------------- #
class AppProcess:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, venv: Path, cwd: Path, entrypoint: str, log):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        cmd = [venv_python(venv), str(cwd / entrypoint)]
        log("Launching application …")
        self.proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, **kwargs)

    def stop(self, timeout: int = 15, log=None):
        if not self.is_running():
            self.proc = None
            return
        if log:
            log(f"Stopping running application (pid {self.proc.pid}) …")
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if log:
                log("App did not exit — forcing stop.")
            try:
                if os.name == "nt":
                    run_cmd(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], check=False)
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                self.proc.kill()
            self.proc.wait()
        self.proc = None


# --------------------------------------------------------------------------- #
# Modern GUI
# --------------------------------------------------------------------------- #
class UpdaterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_TITLE} — {APP_SUBTITLE}")
        self.configure(fg_color=BG)

        # center window
        w, h = 700, 580
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.resizable(False, False)

        self.settings = load_settings()
        self.git = GitRepo(REPO_URL, self.settings["branch"], MIRROR_DIR)
        self.app_proc = AppProcess()
        self._stop_flag = False

        self._build_ui()
        threading.Thread(target=self._update_loop, daemon=True).start()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self._build_brand_bar()
        self._build_status_card()
        self._build_log_card()

    def _build_brand_bar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=28, pady=(24, 4))

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", anchor="w")

        # logo mark
        ctk.CTkLabel(
            left, text="◆",
            width=42, height=42, corner_radius=12,
            fg_color=ACCENT,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white",
        ).pack(side="left", padx=(0, 14))

        titles = ctk.CTkFrame(left, fg_color="transparent")
        titles.pack(side="left", anchor="w")

        ctk.CTkLabel(
            titles, text=APP_TITLE,
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=TEXT, anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            titles, text=APP_SUBTITLE,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=MUTED, anchor="w",
        ).pack(anchor="w")

        # version badge
        ctk.CTkLabel(
            bar, text=f"  {APP_VERSION}  ",
            height=26, corner_radius=13,
            fg_color=BORDER,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=MUTED,
        ).pack(side="right", anchor="e")

    def _build_status_card(self):
        card = ctk.CTkFrame(
            self, fg_color=CARD, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        card.pack(fill="x", padx=28, pady=(18, 12))

        # --- top row: icon + status text --------------------------------- #
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=22, pady=(20, 12))

        self.status_icon = ctk.CTkLabel(
            row, text="◇",
            width=52, height=52, corner_radius=14,
            fg_color=ACCENT,
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white",
        )
        self.status_icon.pack(side="left", padx=(0, 16))

        text_col = ctk.CTkFrame(row, fg_color="transparent")
        text_col.pack(side="left", fill="x", expand=True)

        self.status_label = ctk.CTkLabel(
            text_col, text="Initialising …",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color=TEXT, anchor="w",
        )
        self.status_label.pack(anchor="w")

        self.detail_label = ctk.CTkLabel(
            text_col, text="Preparing updater",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=MUTED, anchor="w",
        )
        self.detail_label.pack(anchor="w", pady=(2, 0))

        # --- progress bar ------------------------------------------------ #
        self.progress = ctk.CTkProgressBar(
            card, height=6, corner_radius=3,
            fg_color=BORDER, progress_color=ACCENT,
        )
        self.progress.pack(fill="x", padx=22, pady=(2, 6))
        self.progress.set(0)

        # --- step indicator --------------------------------------------- #
        steps_row = ctk.CTkFrame(card, fg_color="transparent")
        steps_row.pack(fill="x", padx=22, pady=(6, 20))

        self.step_widgets = {}
        step_defs = [
            ("check",    "CHECK"),
            ("download", "DOWNLOAD"),
            ("install",  "INSTALL"),
            ("launch",   "LAUNCH"),
        ]
        for key, label in step_defs:
            col = ctk.CTkFrame(steps_row, fg_color="transparent")
            col.pack(side="left", expand=True, fill="x")

            dot = ctk.CTkFrame(col, width=12, height=12,
                               corner_radius=6, fg_color=BORDER)
            dot.pack(pady=(0, 6))
            dot.pack_propagate(False)

            lbl = ctk.CTkLabel(
                col, text=label,
                font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                text_color=MUTED,
            )
            lbl.pack()

            self.step_widgets[key] = (dot, lbl)

    def _build_log_card(self):
        card = ctk.CTkFrame(
            self, fg_color=CARD, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        card.pack(fill="both", expand=True, padx=28, pady=(0, 24))

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(14, 6))

        ctk.CTkLabel(
            head, text="ACTIVITY",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=MUTED, anchor="w",
        ).pack(side="left")

        self.log_box = ctk.CTkTextbox(
            card,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=LOG_BG,
            border_width=0,
            corner_radius=10,
            text_color=LOG_TEXT,
            activate_scrollbars=True,
        )
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # UI updates (thread-safe)
    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}]  {msg}\n"

        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.after(0, _append)
        LOG.info(msg)

    def set_status(self, text: str, detail: str = "",
                   progress: float | None = None,
                   icon: str | None = None,
                   tone: str = ACCENT):
        def _update():
            self.status_label.configure(text=text)
            if detail:
                self.detail_label.configure(text=detail)
            if progress is not None:
                self.progress.set(progress)
                self.progress.configure(progress_color=tone)
            if icon is not None:
                self.status_icon.configure(text=icon, fg_color=tone)

        self.after(0, _update)

    def set_step(self, step: str, state: str):
        """state: 'pending' | 'active' | 'done' | 'error'"""
        if step not in self.step_widgets:
            return
        dot, lbl = self.step_widgets[step]
        colors = {
            "pending": (BORDER,  MUTED),
            "active":  (ACCENT,  TEXT),
            "done":    (SUCCESS, TEXT),
            "error":   (ERROR,   TEXT),
        }
        dot_color, text_color = colors.get(state, (BORDER, MUTED))

        def _update():
            dot.configure(fg_color=dot_color)
            lbl.configure(text_color=text_color)

        self.after(0, _update)

    # ------------------------------------------------------------------ #
    # Update loop
    # ------------------------------------------------------------------ #
    def _update_loop(self):
        try:
            # ---- 1. Internet -------------------------------------------------- #
            self.set_step("check", "active")
            self.set_status("Checking connection …", "Verifying network access",
                            progress=0.05, icon="◎", tone=ACCENT)
            self.log("Checking internet connection …")

            if not internet_available():
                self.log("No internet connection detected.")
                self.set_step("check", "done")
                self.set_status("Offline mode", "Starting last installed build",
                                progress=0.5, icon="◆", tone=WARNING)
                time.sleep(0.6)
                self._launch_existing()
                return

            self.log("Internet connection OK.")
            self.set_step("check", "done")

            # ---- 2. Fetch ----------------------------------------------------- #
            self.set_step("download", "active")
            self.set_status("Checking for updates …", "Contacting server",
                            progress=0.15, icon="▼", tone=ACCENT)

            self.git.ensure_cloned(self.log)
            self.git.fetch(self.log)
            sha = self.git.rev()
            self.log(f"Latest build on server: {sha[:8]}")

            state = load_json(STATE_FILE, {})
            deployed = state.get("sha")

            if sha == state.get("bad_sha") and sha != deployed:
                # This commit was installed before and did not start. It is not
                # tried again; a newer commit clears the mark.
                self.log(f"Build {sha[:8]} failed to start earlier — keeping the current build.")
                self.set_step("download", "done")
                self.set_step("install",  "done")
                self.set_status("Update skipped", f"Build {sha[:8]} did not start last time",
                                progress=1.0, icon="◆", tone=WARNING)
                time.sleep(0.6)
                self._launch_existing()
                return

            if sha == deployed:
                self.log("Application is up to date.")
                self.set_step("download", "done")
                self.set_step("install",  "done")
                self.set_step("launch",   "active")
                self.set_status("Up to date", f"Build {sha[:8]} already installed",
                                progress=1.0, icon="✓", tone=SUCCESS)
                time.sleep(0.6)
                self._launch_existing()
                return

            # ---- 3. Download -------------------------------------------------- #
            self.log(f"New build detected: {sha[:8]}")
            self.set_status("Downloading update …", f"New build {sha[:8]}",
                            progress=0.35, icon="▼", tone=ACCENT)

            self.app_proc.stop(log=self.log)
            release_dir = RELEASES_DIR / sha
            if release_dir.exists():
                shutil.rmtree(release_dir, ignore_errors=True)
            self.git.extract(sha, release_dir, self.log)
            self.set_step("download", "done")

            # ---- 4. Install --------------------------------------------------- #
            req_file = release_dir / REQUIREMENTS_FILE
            new_hash = sha256_file(req_file)

            try:
                # A rebuilt environment is empty, whatever the stored hash says.
                rebuilt = ensure_venv(VENV_DIR, self.log)
            except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
                self.log(f"Runtime environment could not be prepared: {exc}")
                self.set_step("install", "error")
                self.set_status("Update failed", "Runtime environment not ready",
                                progress=0.0, icon="✕", tone=ERROR)
                self._launch_existing(exclude=release_dir)
                return
            if rebuilt:
                state.pop("deps_hash", None)
                save_json(STATE_FILE, state)

            if new_hash and new_hash != state.get("deps_hash"):
                self.set_step("install", "active")
                self.set_status("Installing components …", "This may take a few minutes",
                                progress=0.6, icon="◆", tone=ACCENT)
                try:
                    install_requirements(VENV_DIR, req_file, self.log)
                except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
                    self.log(f"Component installation failed: {exc}")
                    self.log("Starting the previous build; the update is retried at the next start.")
                    # The new build is NOT recorded as deployed: its packages are
                    # missing, and recording it would make the next start report
                    # "up to date" and never install them. The stored package
                    # hash is dropped because a failed install can leave the
                    # environment between two sets of packages.
                    state.pop("deps_hash", None)
                    save_json(STATE_FILE, state)
                    self.set_step("install", "error")
                    self.set_status("Update failed", "Component installation failed",
                                    progress=0.0, icon="✕", tone=ERROR)
                    self._launch_existing(exclude=release_dir)
                    return
                state["deps_hash"] = new_hash
            else:
                self.log("Required components unchanged — skipping install.")

            self.set_step("install", "done")

            # ---- 5. Save & prune --------------------------------------------- #
            state["sha"] = sha
            state.pop("bad_sha", None)
            save_json(STATE_FILE, state)
            prune_releases(RELEASES_DIR, KEEP_RELEASES, sha, self.log)

            # ---- 6. Launch ---------------------------------------------------- #
            self.set_step("launch", "active")
            self.set_status("Update complete", f"Build {sha[:8]} ready",
                            progress=0.95, icon="▶", tone=SUCCESS)
            time.sleep(0.6)
            previous = RELEASES_DIR / deployed if deployed else None
            if previous is not None and (previous == release_dir or not previous.is_dir()):
                previous = None
            self._launch_release(release_dir, previous=previous)

        except Exception as exc:
            self.log(f"Unexpected error: {exc}")
            self.set_status("Error", str(exc), progress=0.0,
                            icon="✕", tone=ERROR)
            try:
                self._launch_existing()
            except Exception as fallback:
                # Last resort: never let the worker thread die without a message.
                self.log(f"The installed build could not be started: {fallback}")
                self.set_status("Launch failed", str(fallback), progress=0.0,
                                icon="✕", tone=ERROR)

    # ------------------------------------------------------------------ #
    def _launch_existing(self, exclude: Path | None = None):
        """Start the recorded build, else the newest one on disk.

        ``exclude`` names a build that must not be started: one that was just
        downloaded but whose packages could not be installed.
        """
        if not RELEASES_DIR.exists():
            self.log("No installed builds found — cannot launch.")
            self.set_status("No build available", progress=0.0,
                            icon="✕", tone=ERROR)
            return

        dirs = sorted(
            [d for d in RELEASES_DIR.iterdir() if d.is_dir() and d != exclude],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if not dirs:
            self.log("No installed builds found — cannot launch.")
            self.set_status("No build available", progress=0.0,
                            icon="✕", tone=ERROR)
            return

        state = load_json(STATE_FILE, {})
        sha = state.get("sha")
        recorded = RELEASES_DIR / sha if sha else None
        release_dir = recorded if recorded in dirs else dirs[0]
        self._launch_release(release_dir)

    def _launch_release(self, release_dir: Path, previous: Path | None = None):
        """Start a build. ``previous`` is the build to go back to if this one,
        freshly installed, exits with an error during the start check."""
        try:
            if ensure_venv(VENV_DIR, self.log):
                # The environment was missing or built for another Python: it has
                # no packages, so this build's requirements go in before it starts.
                state = load_json(STATE_FILE, {})
                state.pop("deps_hash", None)
                save_json(STATE_FILE, state)
                req_file = release_dir / REQUIREMENTS_FILE
                if req_file.is_file():
                    self.set_status("Installing components …", "This may take a few minutes",
                                    progress=0.6, icon="◆", tone=ACCENT)
                    install_requirements(VENV_DIR, req_file, self.log)
                    state["deps_hash"] = sha256_file(req_file)
                    save_json(STATE_FILE, state)
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            # This runs on the worker thread, often as the fallback after another
            # failure. Anything escaping here would end the thread and leave the
            # window on its previous message, so every failure is shown instead.
            self.log(f"Runtime environment could not be prepared: {exc}")
            self.set_step("launch", "error")
            self.set_status("Launch failed", "Runtime environment not ready",
                            progress=0.0, icon="✕", tone=ERROR)
            return

        entry = release_dir / ENTRYPOINT
        if not entry.exists():
            self.log(f"Application entry point missing in build {release_dir.name[:8]}.")
            self.set_status("Launch failed", "Application files not found",
                            progress=0.0, icon="✕", tone=ERROR)
            return

        self.set_step("launch", "done")
        self.set_status("Starting application …", "Closing updater shortly",
                        progress=1.0, icon="▶", tone=SUCCESS)
        try:
            self.app_proc.start(VENV_DIR, release_dir, ENTRYPOINT, self.log)
        except OSError as exc:
            self.log(f"Application could not be started: {exc}")
            self.set_step("launch", "error")
            self.set_status("Launch failed", "Application could not be started",
                            progress=0.0, icon="✕", tone=ERROR)
            return
        seconds = getattr(self, "settings", {}).get("start_check_seconds", 0)
        if previous is not None and seconds > 0 and self._failed_to_start(seconds):
            self._roll_back(release_dir, previous)
            return
        self.after(1500, self.destroy)

    def _failed_to_start(self, seconds: float) -> bool:
        """True when the application just started exits with an error in time."""
        self.set_status("Checking the new build …", f"Watching it for {seconds:.0f} s",
                        progress=1.0, icon="▶", tone=ACCENT)
        try:
            code = self.app_proc.proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            return False          # still running: it started
        return code != 0          # 0 is the operator closing it

    def _roll_back(self, failed: Path, previous: Path):
        self.log(f"Build {failed.name[:8]} exited with an error right after starting.")
        self.log(f"Going back to build {previous.name[:8]}.")
        state = load_json(STATE_FILE, {})
        state["sha"] = previous.name
        state["bad_sha"] = failed.name
        # The environment now holds the failed build's packages.
        state.pop("deps_hash", None)
        save_json(STATE_FILE, state)
        self.app_proc.proc = None
        req_file = previous / REQUIREMENTS_FILE
        try:
            if req_file.is_file() and sha256_file(req_file) != sha256_file(failed / REQUIREMENTS_FILE):
                install_requirements(VENV_DIR, req_file, self.log)
            state["deps_hash"] = sha256_file(req_file)
            save_json(STATE_FILE, state)
        except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
            self.log(f"Previous components could not be restored: {exc}")
        self.set_status("Update rolled back", f"Build {failed.name[:8]} did not start",
                        progress=1.0, icon="◆", tone=WARNING)
        self._launch_release(previous)

    # ------------------------------------------------------------------ #
    def on_close(self):
        self._stop_flag = True
        self.app_proc.stop(log=self.log)
        self.destroy()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def main():
    app = UpdaterApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()