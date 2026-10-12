"""Settings changed on this PC survive an update that replaces config.json."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import machine_settings
import startup_settings

DEFAULTS = {
    'camera_count': 1,
    'cameras': [{'index': 0, 'rotation': 180}, {'index': 2, 'rotation': 0}],
    'inspection': {'strip_width_mm': 4.0, 'strip_width_tolerance_mm': 1.0, 'sizes': ['S', 'M']},
    'local_inspection': {'offset_pixels': 100.0},
}


class MachineSettingsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.build = Path(folder.name) / 'build'
        self.build.mkdir()
        self.defaults = self.build / 'config.json'
        self.defaults.write_text(json.dumps(DEFAULTS))
        self.machine = Path(folder.name) / 'documents' / 'Files' / 'machine_settings.json'

    def changed(self):
        config = json.loads(json.dumps(DEFAULTS))
        config['cameras'][0]['index'] = 3
        config['inspection']['strip_width_tolerance_mm'] = 0.5
        return config

    def test_nothing_saved_gives_the_defaults(self):
        self.assertEqual(machine_settings.load_config(self.defaults, self.machine), DEFAULTS)

    def test_only_the_differences_are_stored(self):
        machine_settings.save_config(self.changed(), self.defaults, self.machine)
        self.assertEqual(json.loads(self.machine.read_text()),
                         {'cameras': [{'index': 3}, {}], 'inspection': {'strip_width_tolerance_mm': 0.5}})
        self.assertEqual(json.loads(self.defaults.read_text()), DEFAULTS)

    def test_saved_settings_come_back(self):
        machine_settings.save_config(self.changed(), self.defaults, self.machine)
        self.assertEqual(machine_settings.load_config(self.defaults, self.machine), self.changed())

    def test_an_update_keeps_local_changes_and_brings_new_defaults(self):
        machine_settings.save_config(self.changed(), self.defaults, self.machine)
        newer = json.loads(json.dumps(DEFAULTS))
        newer['local_inspection']['offset_pixels'] = 120.0        # a default the update changed
        newer['inspection']['end_exclusion_percent'] = 5.0        # a setting the update added
        newer['cameras'][0]['fps'] = 10
        del newer['camera_count']                                  # a setting the update removed
        update = self.build.parent / 'newer' / 'config.json'
        update.parent.mkdir()
        update.write_text(json.dumps(newer))
        config = machine_settings.load_config(update, self.machine)
        self.assertEqual(config['cameras'][0], {'index': 3, 'rotation': 180, 'fps': 10})
        self.assertEqual(config['inspection']['strip_width_tolerance_mm'], 0.5)
        self.assertEqual(config['inspection']['end_exclusion_percent'], 5.0)
        self.assertEqual(config['local_inspection']['offset_pixels'], 120.0)
        self.assertNotIn('camera_count', config)

    def test_setting_a_value_back_to_the_default_removes_it(self):
        machine_settings.save_config(self.changed(), self.defaults, self.machine)
        machine_settings.save_config(DEFAULTS, self.defaults, self.machine)
        self.assertEqual(json.loads(self.machine.read_text()), {})

    def test_a_changed_list_of_plain_values_replaces_the_default(self):
        config = json.loads(json.dumps(DEFAULTS))
        config['inspection']['sizes'] = ['S', 'M', 'L']
        machine_settings.save_config(config, self.defaults, self.machine)
        self.assertEqual(machine_settings.load_config(self.defaults, self.machine)['inspection']['sizes'],
                         ['S', 'M', 'L'])

    def test_an_unreadable_file_gives_the_defaults(self):
        self.machine.parent.mkdir(parents=True)
        for text in ('not json', '[1, 2]'):
            self.machine.write_text(text)
            self.assertEqual(machine_settings.load_config(self.defaults, self.machine), DEFAULTS)

    def test_changes_that_are_no_longer_valid_fall_back_to_the_defaults(self):
        machine_settings.save_config(self.changed(), self.defaults, self.machine)
        def validate(config):
            if config['cameras'][0]['index'] == 3:
                raise ValueError('camera 3 is not allowed any more')
        self.assertEqual(machine_settings.load_config(self.defaults, self.machine, validate), DEFAULTS)
        self.assertTrue(self.machine.exists())

    def test_the_startup_form_reads_and_saves_through_machine_settings(self):
        with patch.object(startup_settings, 'CONFIG_PATH', self.defaults), \
                patch.object(machine_settings, 'machine_path', return_value=self.machine), \
                patch('config_validation.validate_config'):
            startup_settings.save_config(self.changed())
            self.assertEqual(startup_settings.load_config(), self.changed())
        self.assertEqual(json.loads(self.defaults.read_text()), DEFAULTS)


if __name__ == '__main__':
    unittest.main()
