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
        self.ensure = patch.object(self.updater, 'ensure_venv').start()
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


if __name__ == '__main__':
    unittest.main()
