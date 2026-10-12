"""Updater decisions with git, pip, the window and the app all replaced by stand-ins."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_updater():
    """Import updater/updater.py without CustomTkinter or a display."""
    toolkit = types.ModuleType('customtkinter')
    toolkit.CTk = type('CTk', (), {})
    toolkit.set_appearance_mode = toolkit.set_default_color_theme = lambda *a, **k: None
    with patch.dict(sys.modules, {'customtkinter': toolkit}):
        spec = importlib.util.spec_from_file_location('_updater_under_test', ROOT / 'updater' / 'updater.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class UpdaterCase(unittest.TestCase):
    """A starter with no window, pointed at a temporary folder."""

    SHA_OLD, SHA_NEW = 'a' * 40, 'b' * 40

    def setUp(self):
        self.updater = load_updater()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        for name, value in (('RELEASES_DIR', root / 'releases'), ('VENV_DIR', root / 'venv'),
                            ('STATE_FILE', root / 'state.json'), ('MIRROR_DIR', root / 'repo.git')):
            patcher = patch.object(self.updater, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('internet_available',):
            patcher = patch.object(self.updater, name, return_value=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.updater.time, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = self.updater.UpdaterApp.__new__(self.updater.UpdaterApp)
        self.app.log = MagicMock()
        self.app.set_status = MagicMock()
        self.app.set_step = MagicMock()
        self.app.after = MagicMock()
        self.app.destroy = MagicMock()
        self.app.app_proc = MagicMock()
        self.app.settings = {'branch': 'main', 'start_check_seconds': 0}
        self.app.git = MagicMock()
        self.app.git.rev.return_value = self.SHA_NEW
        self.app.git.extract.side_effect = self.extract

    def extract(self, sha, destination, log):
        destination.mkdir(parents=True)
        (destination / 'main_1366.py').write_text('print(1)')
        (destination / 'requirements.txt').write_text('numpy')

    def release(self, sha, requirements='numpy'):
        folder = self.updater.RELEASES_DIR / sha
        folder.mkdir(parents=True)
        (folder / 'main_1366.py').write_text('print(1)')
        (folder / 'requirements.txt').write_text(requirements)
        return folder

    def state(self):
        return json.loads(self.updater.STATE_FILE.read_text())

    def started_from(self):
        return self.app.app_proc.start.call_args.args[1]


class FailedInstallTests(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.release(self.SHA_OLD, requirements='old')
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD, 'deps_hash': 'old-hash'})
        self.ensure = patch.object(self.updater, 'ensure_venv', return_value=False).start()
        self.addCleanup(patch.stopall)

    def test_a_failed_install_is_not_recorded_and_the_previous_build_starts(self):
        failure = subprocess.CalledProcessError(1, ['pip'])
        with patch.object(self.updater, 'install_requirements', side_effect=failure):
            self.app._update_loop()
        self.assertEqual(self.state()['sha'], self.SHA_OLD)
        self.assertNotIn('deps_hash', self.state())
        self.assertEqual(self.started_from().name, self.SHA_OLD)

    def test_the_install_is_tried_again_at_the_next_start(self):
        failure = subprocess.CalledProcessError(1, ['pip'])
        with patch.object(self.updater, 'install_requirements', side_effect=failure):
            self.app._update_loop()
        with patch.object(self.updater, 'install_requirements') as install:
            self.app._update_loop()
        install.assert_called_once()
        self.assertEqual(self.state()['sha'], self.SHA_NEW)
        self.assertEqual(self.started_from().name, self.SHA_NEW)

    def test_a_first_install_that_fails_starts_nothing(self):
        self.updater.STATE_FILE.unlink()
        import shutil
        shutil.rmtree(self.updater.RELEASES_DIR / self.SHA_OLD)
        failure = subprocess.CalledProcessError(1, ['pip'])
        with patch.object(self.updater, 'install_requirements', side_effect=failure):
            self.app._update_loop()
        self.app.app_proc.start.assert_not_called()


class EmptyEnvironmentTests(UpdaterCase):
    """An environment that was just built has no packages, whatever the state file says."""

    def setUp(self):
        super().setUp()
        self.folder_old = self.release(self.SHA_OLD)
        self.hash = self.updater.sha256_file(self.folder_old / 'requirements.txt')
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD, 'deps_hash': self.hash})
        self.addCleanup(patch.stopall)

    def test_starting_an_installed_build_fills_a_rebuilt_environment(self):
        self.app.git.rev.return_value = self.SHA_OLD
        with patch.object(self.updater, 'ensure_venv', return_value=True), \
                patch.object(self.updater, 'install_requirements') as install:
            self.app._update_loop()
        install.assert_called_once()
        self.assertEqual(install.call_args.args[1], self.folder_old / 'requirements.txt')
        self.assertEqual(self.state()['deps_hash'], self.hash)
        self.assertEqual(self.started_from().name, self.SHA_OLD)

    def test_an_existing_environment_is_not_reinstalled(self):
        self.app.git.rev.return_value = self.SHA_OLD
        with patch.object(self.updater, 'ensure_venv', return_value=False), \
                patch.object(self.updater, 'install_requirements') as install:
            self.app._update_loop()
        install.assert_not_called()

    def test_an_update_with_unchanged_requirements_still_fills_a_rebuilt_environment(self):
        with patch.object(self.updater, 'ensure_venv', side_effect=[True, False]), \
                patch.object(self.updater, 'install_requirements') as install:
            self.app._update_loop()
        install.assert_called_once()
        self.assertEqual(self.state()['sha'], self.SHA_NEW)

    def test_offline_start_fills_a_rebuilt_environment(self):
        with patch.object(self.updater, 'internet_available', return_value=False), \
                patch.object(self.updater, 'ensure_venv', return_value=True), \
                patch.object(self.updater, 'install_requirements') as install:
            self.app._update_loop()
        install.assert_called_once()


class FallbackLaunchTests(UpdaterCase):
    """A failure while starting the installed build is shown, never left to end the thread."""

    def setUp(self):
        super().setUp()
        self.release(self.SHA_OLD)
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD})
        self.app.git.fetch.side_effect = subprocess.CalledProcessError(128, ['git'])
        self.addCleanup(patch.stopall)

    def shown(self):
        return [call.args[0] for call in self.app.set_status.call_args_list]

    def test_a_failing_environment_in_the_fallback_is_reported(self):
        for failure in (subprocess.CalledProcessError(1, ['python', '-m', 'venv']),
                        subprocess.TimeoutExpired(['pip'], 5), OSError('disk full'),
                        RuntimeError('Python 3.12 was not found')):
            with self.subTest(failure=type(failure).__name__):
                self.app.set_status.reset_mock()
                with patch.object(self.updater, 'ensure_venv', side_effect=failure):
                    self.app._update_loop()
                self.assertEqual(self.shown()[-1], 'Launch failed')
                self.app.app_proc.start.assert_not_called()

    def test_an_app_that_cannot_be_started_is_reported(self):
        self.app.app_proc.start.side_effect = OSError('python.exe missing')
        with patch.object(self.updater, 'ensure_venv', return_value=False):
            self.app._update_loop()
        self.assertEqual(self.shown()[-1], 'Launch failed')
        self.app.after.assert_not_called()

    def test_a_fetch_failure_still_starts_the_installed_build(self):
        with patch.object(self.updater, 'ensure_venv', return_value=False):
            self.app._update_loop()
        self.assertEqual(self.started_from().name, self.SHA_OLD)


class TimeLimitTests(UpdaterCase):
    """No external command may wait for ever."""

    def commands(self, action):
        seen = []
        def record(cmd, **kw):
            seen.append((list(map(str, cmd)), kw))
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        with patch.object(self.updater.subprocess, 'run', side_effect=record):
            action()
        return seen

    def test_every_git_and_pip_command_has_a_time_limit_and_cannot_prompt(self):
        repo = self.updater.GitRepo('https://example.invalid/x.git', 'main', self.updater.MIRROR_DIR)
        seen = self.commands(lambda: (repo.ensure_cloned(lambda m: None), repo.fetch(lambda m: None), repo.rev(),
                                      self.updater.install_requirements(self.updater.VENV_DIR, Path('r.txt'), lambda m: None)))
        self.assertEqual(len(seen), 4)
        for cmd, kw in seen:
            self.assertGreater(kw.get('timeout') or 0, 0, cmd)
        for cmd, kw in seen[:3]:
            self.assertEqual(kw['env']['GIT_TERMINAL_PROMPT'], '0', cmd)
        self.assertIn('--no-input', seen[3][0])

    def test_a_clone_that_fails_leaves_no_half_made_mirror(self):
        repo = self.updater.GitRepo('https://example.invalid/x.git', 'main', self.updater.MIRROR_DIR)
        def stall(cmd, **kw):
            self.updater.MIRROR_DIR.mkdir()
            raise subprocess.TimeoutExpired(cmd, kw['timeout'])
        with patch.object(self.updater.subprocess, 'run', side_effect=stall):
            with self.assertRaises(subprocess.TimeoutExpired):
                repo.ensure_cloned(lambda m: None)
        self.assertFalse(self.updater.MIRROR_DIR.exists())

    def test_a_stalled_fetch_starts_the_installed_build(self):
        self.release(self.SHA_OLD)
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD})
        self.app.git.fetch.side_effect = subprocess.TimeoutExpired(['git', 'fetch'], 300)
        with patch.object(self.updater, 'ensure_venv', return_value=False):
            self.app._update_loop()
        self.assertEqual(self.started_from().name, self.SHA_OLD)
        self.assertEqual(self.state()['sha'], self.SHA_OLD)

    def test_a_stalled_install_keeps_the_previous_build(self):
        self.release(self.SHA_OLD, requirements='old')
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD, 'deps_hash': 'x'})
        with patch.object(self.updater, 'ensure_venv', return_value=False), \
                patch.object(self.updater, 'install_requirements',
                             side_effect=subprocess.TimeoutExpired(['pip'], 3600)):
            self.app._update_loop()
        self.assertEqual(self.state()['sha'], self.SHA_OLD)
        self.assertEqual(self.started_from().name, self.SHA_OLD)


class SettingsTests(UpdaterCase):
    def read(self, text):
        path = Path(self.folder.name) / 'updater.json'
        path.write_text(text)
        return self.updater.load_settings(path)

    def test_no_file_keeps_main_and_no_start_check(self):
        self.assertEqual(self.updater.load_settings(Path(self.folder.name) / 'absent.json'),
                         {'branch': 'main', 'start_check_seconds': 0.0})

    def test_branch_and_start_check_are_read(self):
        self.assertEqual(self.read('{"branch": " release ", "start_check_seconds": 20}'),
                         {'branch': 'release', 'start_check_seconds': 20.0})

    def test_bad_values_fall_back_to_the_defaults(self):
        for text in ('not json', '[]', '{"branch": "", "start_check_seconds": -1}',
                     '{"branch": 5, "start_check_seconds": true}', '{"start_check_seconds": "soon"}'):
            with self.subTest(text=text):
                self.assertEqual(self.read(text), {'branch': 'main', 'start_check_seconds': 0.0})


class StartCheckTests(UpdaterCase):
    """After an update, a build that exits with an error at once is rolled back."""

    def setUp(self):
        super().setUp()
        self.release(self.SHA_OLD)
        self.updater.save_json(self.updater.STATE_FILE, {'sha': self.SHA_OLD, 'deps_hash': 'x'})
        self.app.settings = {'branch': 'main', 'start_check_seconds': 20}
        patch.object(self.updater, 'ensure_venv', return_value=False).start()
        self.install = patch.object(self.updater, 'install_requirements').start()
        self.addCleanup(patch.stopall)

    def exits(self, code):
        self.app.app_proc.proc = MagicMock()   # the real start() makes a new process
        if code is None:
            self.app.app_proc.proc.wait.side_effect = subprocess.TimeoutExpired(['app'], 20)
        else:
            self.app.app_proc.proc.wait.return_value = code

    def starts(self):
        return [call.args[1].name for call in self.app.app_proc.start.call_args_list]

    def test_a_build_that_keeps_running_is_kept(self):
        self.exits(None)
        self.app._update_loop()
        self.assertEqual(self.starts(), [self.SHA_NEW])
        self.assertEqual(self.state()['sha'], self.SHA_NEW)
        self.assertNotIn('bad_sha', self.state())

    def test_a_build_closed_normally_is_kept(self):
        self.exits(0)
        self.app._update_loop()
        self.assertEqual(self.starts(), [self.SHA_NEW])
        self.assertEqual(self.state()['sha'], self.SHA_NEW)

    def test_a_build_that_exits_with_an_error_is_rolled_back(self):
        self.exits(1)
        self.app._update_loop()
        self.assertEqual(self.starts(), [self.SHA_NEW, self.SHA_OLD])
        self.assertEqual(self.state()['sha'], self.SHA_OLD)
        self.assertEqual(self.state()['bad_sha'], self.SHA_NEW)

    def test_the_bad_commit_is_not_installed_again_but_a_newer_one_is(self):
        self.exits(1)
        self.app._update_loop()
        self.app.app_proc.start.reset_mock()
        self.app.git.extract.reset_mock()
        self.app._update_loop()
        self.app.git.extract.assert_not_called()
        self.assertEqual(self.starts(), [self.SHA_OLD])
        newer = 'c' * 40
        self.app.git.rev.return_value = newer
        self.exits(None)
        self.app.app_proc.start.reset_mock()
        self.app._update_loop()
        self.assertEqual(self.starts(), [newer])
        self.assertEqual(self.state()['sha'], newer)
        self.assertNotIn('bad_sha', self.state())

    def test_without_the_setting_nothing_is_watched(self):
        self.app.settings = {'branch': 'main', 'start_check_seconds': 0}
        self.app._update_loop()
        self.app.app_proc.proc.wait.assert_not_called()
        self.assertEqual(self.starts(), [self.SHA_NEW])

    def test_an_ordinary_start_is_never_watched(self):
        self.app.git.rev.return_value = self.SHA_OLD
        self.app._update_loop()
        self.app.app_proc.proc.wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
