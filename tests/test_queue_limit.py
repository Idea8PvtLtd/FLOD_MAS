"""auto_trigger.max_queued_captures: the capture queue cap for small-memory devices."""
import copy
import json
from pathlib import Path
import unittest

from auto_trigger.controller import AutoController
from config_validation import validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config.json').read_text())


def controller(limit):
    config = copy.deepcopy(CONFIG)
    if limit is None:
        config['auto_trigger'].pop('max_queued_captures', None)
    else:
        config['auto_trigger']['max_queued_captures'] = limit
    return AutoController(config, lambda: None)


class QueueLimitTests(unittest.TestCase):
    def test_no_setting_or_zero_means_no_limit(self):
        for limit in (None, 0):
            control = controller(limit)
            control.preparing_count = 5
            control.pending_jobs.extend([{}] * 5)
            self.assertFalse(control._queue_full())

    def test_checking_and_waiting_captures_count_towards_the_limit(self):
        control = controller(1)
        self.assertFalse(control._queue_full())
        control.preparing_count = 1
        self.assertTrue(control._queue_full())
        control.preparing_count = 0
        control.pending_jobs.append({})
        self.assertTrue(control._queue_full())

    def test_the_capture_being_inspected_is_not_counted(self):
        control = controller(1)
        control.future = object()
        self.assertFalse(control._queue_full())

    def test_validation(self):
        for good in (0, 1, 3):
            config = copy.deepcopy(CONFIG)
            config['auto_trigger']['max_queued_captures'] = good
            validate_config(config)
        for bad in (-1, 1.5, True, '1'):
            config = copy.deepcopy(CONFIG)
            config['auto_trigger']['max_queued_captures'] = bad
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
