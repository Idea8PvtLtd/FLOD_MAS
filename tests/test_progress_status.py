"""The progress line never keeps a busy text once the work has ended."""
from types import SimpleNamespace
import unittest

from inspection.dashboard import AutomaticDashboard


class Label:
    text = ''

    def configure(self, text=None, **_):
        if text is not None:
            self.text = text
    config = configure


class Controller:
    generation = 0
    queue_text = 'Queue: 0 checking | 0 waiting | 0 inspecting'
    future = None
    gate_busy = False
    reset_requested = False
    paused = False

    def __init__(self):
        self.busy = False
        self.events = []

    def set_active(self, active):
        pass

    def poll(self):
        events, self.events = self.events, []
        return events


class App(AutomaticDashboard):
    def __init__(self):
        self.root = SimpleNamespace(after=lambda delay, callback: None, update_idletasks=lambda: None)
        self.auto_controller = Controller()
        self.video_paused, self.video_streaming = False, True
        self.progress_label, self.inspection_queue_label = Label(), Label()
        self.progress_bar = {}
        self._reset_due = None

    def _set_activity_animation(self, busy):
        pass

    def update_progress(self, value, text):
        self.progress_label.configure(text=text)
        self._progress_text = text


class ProgressStatusTests(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.controller = self.app.auto_controller

    def poll(self, busy, *events):
        self.controller.busy = busy
        self.controller.future = object() if busy else None
        self.controller.events = list(events)
        self.app._poll_automatic()
        return self.app.progress_label.text

    def test_the_busy_text_shows_only_while_busy(self):
        self.assertEqual(self.poll(True, ('status', 0, 'Inspecting captured fabric…')),
                         'Inspecting captured fabric…')
        # Seen on the mini PC: work over, queue empty, yet "Inspecting..." stayed.
        self.assertEqual(self.poll(False), 'Ready — waiting for a hand cycle')

    def test_a_message_hidden_while_busy_is_shown_afterwards(self):
        self.poll(True, ('status', 0, 'Fabric check: no_fabric (91%)'))
        self.assertEqual(self.app.progress_label.text, 'Inspecting captured fabric…')
        self.assertEqual(self.poll(False), 'Fabric check: no_fabric (91%)')

    def test_an_idle_message_is_left_alone(self):
        self.poll(False, ('status', 0, 'Hand present — waiting for hands to leave'))
        self.assertEqual(self.poll(False), 'Hand present — waiting for hands to leave')


if __name__ == '__main__':
    unittest.main()
