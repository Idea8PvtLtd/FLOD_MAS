"""Scene mode: take a key photo when the picture has become still and has changed.

The hand mode (checkpoint.py) waits for a hand to appear and leave. This mode does not
need to see the hand at all: it watches how much of the picture moves from one frame to
the next. While a hand or the board moves, a large share of the picture changes; once
everything is at rest almost nothing does. A photo is offered when the picture has been
at rest for a moment AND it differs from the last key photo (a new cloth was laid).

Two small picture functions (the only place cv2 is used) feed a pure state machine:

    small, motion = scene_signals(frame, previous)
    event = monitor.update(motion, bare, small, now)
    if event == 'capture':      # then exactly one of
        monitor.accept()        #   the photo was kept: it is the new key photo
        monitor.reject(now)     #   a hand was still in it: it is offered again later
    elif event == 'cycle_end':  # the board left the bed after at least one key photo
        ...
"""
import cv2
import numpy as np


def scene_signals(frame, previous=None, width=320, level=12):
    """-> (small, motion): small = uint8 grey 2-D picture about `width` wide, cheaply made
    (slicing, no 16 MP resize) and lightly blurred; motion = share (0..1) of its pixels that
    differ from `previous` (a `small` from the call before) by more than `level`; 0.0 when
    previous is None or of another shape."""
    # Every k-th pixel of every k-th row: this touches about 80 thousand of the 16 million
    # pixels, so it costs well under a millisecond where a true resize costs tens.
    step = max(1, frame.shape[1] // max(1, int(width)))
    small = np.ascontiguousarray(frame[::step, ::step])
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Picking single pixels keeps the sensor noise of each one; the blur averages it away
    # so that a resting picture really reads as (almost) no change.
    small = cv2.GaussianBlur(small, (5, 5), 0)
    motion = 0.0
    if previous is not None and previous.shape == small.shape:
        motion = float(np.count_nonzero(cv2.absdiff(small, previous) > level)) / small.size
    return small, motion


def changed_share(small, key, level=12):
    """Share (0..1) of pixels that differ between two `small` pictures after removing their
    mean brightness difference (the camera's exposure drifts)."""
    if small is None or key is None or small.shape != key.shape or small.size == 0:
        return 1.0      # nothing comparable: count it as wholly changed
    difference = small.astype(np.float32) - key.astype(np.float32)
    difference -= float(difference.mean())
    return float(np.count_nonzero(np.abs(difference) > level)) / small.size


# Clock values are sums of frame periods (0.1 + 0.1 + ...), which land a hair under the
# round number; without this allowance a wait of 1.0 s would sometimes take one frame more.
_TICK = 1e-6


class SceneMonitor:
    """Decides, frame by frame, when a key photo is due and when the board has left.

    States:
      BARE      the bed is seen bare (its markers are visible); nothing is on it
      MOVING    something in the picture moves (hand, arm, board)
      SETTLING  the picture is at rest; confirming that it stays so
      STILL     the picture has been at rest long enough; a photo was offered if due
      VETOED    the offered photo was refused (a hand was in it); it is offered again
                after a wait, a limited number of times, while the picture stays at rest

    The defaults were set on the two recordings in data/procedure_videos (10 pictures a
    second): at rest 0.0-0.9 % of the picture changes between frames, 2 % and more while
    anything is handled; the same scene differs from its key photo by up to 2.9 % over
    ten seconds (small shifts, exposure), a newly laid cloth by 12.9 % or more.
    """

    BARE = 'BARE'
    MOVING = 'MOVING'
    SETTLING = 'SETTLING'
    STILL = 'STILL'
    VETOED = 'VETOED'
    STATES = (BARE, MOVING, SETTLING, STILL, VETOED)

    def __init__(self, still_motion=0.01, still_seconds=1.0, change=0.05, bare_seconds=0.5,
                 veto_retry_seconds=1.0, veto_retries=3, level=12):
        self.still_motion = max(0.0, float(still_motion))
        self.still_seconds = max(0.0, float(still_seconds))
        self.change = max(0.0, float(change))
        self.bare_seconds = max(0.0, float(bare_seconds))
        self.veto_retry_seconds = max(0.0, float(veto_retry_seconds))
        self.veto_retries = max(0, int(veto_retries))
        self.level = level
        self.reset()

    @classmethod
    def from_settings(cls, settings):
        """Build from the config's `auto_trigger` section; its `scene` part is optional.
        The file holds percentages, the monitor works with shares (0..1)."""
        settings = settings or {}
        scene = settings.get('scene') or {}
        return cls(
            still_motion=float(scene.get('still_motion_percent', 1.0)) / 100.0,
            still_seconds=float(scene.get('still_seconds', 1.0)),
            change=float(scene.get('change_percent', 5.0)) / 100.0,
            bare_seconds=float(settings.get('marker_confirm_seconds', 0.5)),
            veto_retry_seconds=float(scene.get('veto_retry_seconds', 1.0)),
            veto_retries=int(scene.get('veto_retries', 3)),
        )

    def reset(self):
        """Back to a clean start (pause, new epoch): no key photo, no cycle, no candidate."""
        self.state = self.MOVING     # unknown until the picture has shown what it is
        self.photos = 0              # key photos accepted in this cycle
        self.calm_seconds = 0.0      # how long the picture has been at rest
        self.last_change = None      # share changed against the key photo, last time asked
        self.retries = 0             # offers repeated after a veto, in a row
        self._key = None
        self._candidate = None       # the offered picture, until accept() or reject()
        self._calm_since = None
        self._bare_since = None
        self._veto_at = 0.0

    @property
    def waiting(self):
        """True while an offered photo has been neither accepted nor rejected."""
        return self._candidate is not None

    def update(self, motion, bare, small, now):
        """One frame. `bare`: the bed is seen bare (markers visible). Returns None,
        'capture' (this frame is a candidate key photo: caller then calls accept() or
        reject(now)) or 'cycle_end' (the board left after at least one accepted photo)."""
        if bare:
            # A bare bed is never a photo, and a frame of it breaks the rest period, so a
            # marker reading that flickers cannot let a photo of the empty bed through.
            self._calm_since = None
            self.calm_seconds = 0.0
            if self.state == self.BARE:
                return None
            if self._bare_since is None:
                self._bare_since = now
            if now - self._bare_since >= self.bare_seconds - _TICK:
                return self._enter_bare()
            self._to_moving()
            return None
        self._bare_since = None

        if motion > self.still_motion:
            self._to_moving()
            self._calm_since = None
            self.calm_seconds = 0.0
            return None

        if self._calm_since is None:
            self._calm_since = now
        self.calm_seconds = now - self._calm_since
        if self.state in (self.MOVING, self.BARE):
            self.state = self.SETTLING

        if self.state == self.SETTLING:
            if self.calm_seconds >= self.still_seconds - _TICK:
                self.state = self.STILL
                if self._key is not None:
                    self.last_change = changed_share(small, self._key, self.level)
                    if self.last_change < self.change:
                        return None      # same picture as the key photo: nothing new
                return self._offer(small)
        elif self.state == self.VETOED:
            if (self.retries < self.veto_retries
                    and now - self._veto_at >= self.veto_retry_seconds - _TICK):
                self.retries += 1
                self.state = self.STILL
                return self._offer(small)
        return None

    def accept(self):
        """The candidate became the key photo. False when there was no candidate any more
        (it was dropped because the picture moved again before the answer came)."""
        if self._candidate is None:
            return False
        self._key = self._candidate
        self._candidate = None
        self.photos += 1
        self.retries = 0
        self.last_change = 0.0
        return True

    def reject(self, now):
        """A hand was in the candidate: keep the key photo and offer again later. False
        when there was no candidate any more."""
        if self._candidate is None:
            return False
        self._candidate = None
        self.state = self.VETOED
        self._veto_at = now
        return True

    def describe(self):
        """One line for the status text and the log."""
        change = '-' if self.last_change is None else f'{self.last_change * 100:.1f}%'
        text = (f'{self.state} calm {self.calm_seconds:.1f}s change {change} '
                f'photos {self.photos}')
        if self.state == self.VETOED or self.retries:
            text += f' retry {self.retries}/{self.veto_retries}'
        if self.waiting:
            text += ' candidate'
        return text

    def _offer(self, small):
        self._candidate = small
        return 'capture'

    def _to_moving(self):
        # Movement ends any open question: an unanswered candidate is dropped, the veto
        # count starts again, and the next rest period is judged by the normal rule.
        self.state = self.MOVING
        self._candidate = None
        self.retries = 0

    def _enter_bare(self):
        ended = self.photos > 0
        self.state = self.BARE
        self.photos = 0
        self.retries = 0
        self.last_change = None
        self._key = None
        self._candidate = None
        return 'cycle_end' if ended else None
