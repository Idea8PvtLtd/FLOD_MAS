"""Latest-frame trigger and bed monitor with a FIFO inspection queue.

    submit(raw_frame, sequence, undistorter, profile) is camera-independent.
    poll() returns events; the caller alone owns Tk widgets.
"""
from concurrent.futures import Future
from collections import deque
from .bed_markers import BedMarkers
from datetime import datetime
import inspect
import logging
import queue
import threading
import time
from .checkpoint import Checkpoint, Debouncer
from .models import HandPresence, FabricGate


log = logging.getLogger('flod.trigger')


class _Events(queue.SimpleQueue):
    """The queue the window reads, with every event except a measured line also logged.

    Measured lines are logged where they are measured (inspection/pipeline.py),
    with their figures; here they would only be a name.
    """
    LEVELS = {'error': logging.ERROR, 'startup_error': logging.ERROR, 'line_failed': logging.WARNING}

    def put(self, item, *args, **kwargs):
        event, generation, payload = item
        # A board result carries pictures; it is logged where it is made, like a line.
        if event not in ('result', 'board'):
            log.log(self.LEVELS.get(event, logging.INFO), '%s: %s', event, payload)
        super().put(item, *args, **kwargs)


class _NoGate:
    """Scene mode has no fabric classifier: whether glue lines are there is for the
    line finder to say. This stands in so that the capture queue runs unchanged."""

    def classify(self, frame):
        return {'label': 'not_checked', 'confidence': 1.0, 'accepted': True, 'empty': False, 'scores': {}}


class AutoController(threading.Thread):
    # auto_trigger.mode. 'hand' (the default): a capture when a hand has come and gone,
    # as before. 'scene': a capture of the whole board whenever the picture is still
    # and has changed, with every line shown together; see auto_trigger/scene_monitor.py.
    # Class-level so that instances built without __init__ (the tests) are in hand mode.
    mode = 'hand'
    scene = None          # the SceneMonitor, in scene mode
    board = None          # the lines known on the board now on the bed (inspection.board_lines)
    scene_settings = {}

    def __init__(self, config, pipeline_factory, hand_factory=HandPresence, fabric_factory=FabricGate):
        super().__init__(name='automatic-hand-monitor', daemon=True)
        self.config = config
        self.settings = config['auto_trigger']
        self.mode = self.settings.get('mode', 'hand')
        self.scene_settings = self.settings.get('scene', {})
        self.pipeline_factory = pipeline_factory
        self.hand_factory, self.fabric_factory = hand_factory, fabric_factory
        self.events = _Events()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopped = threading.Event()
        self.active = True
        self.generation = 0
        self.epoch = 0
        self.latest = None
        self.future = None
        self.pending_jobs = deque()
        self.paused = False
        self.pause_revision = 0
        self.has_results = False
        # Line positions measured since the bed was last reset, counted from the
        # left. Failure to measure a line leaves it out, so that line is offered
        # again on the next trigger instead of being counted as done.
        self.inspected_lines = set()
        self.reset_requested = False
        self.reset_ready = False
        self.marker_monitor = None
        self.pipeline = None
        self.ready = False
        self.gate_busy = False
        self.capture_jobs = queue.Queue()
        self.prepared_jobs = queue.SimpleQueue()
        self.preparing_count = 0
        self.job_generation = -1
        self.last_status = None
        # Latest preview geometry, read by the Tk thread. The whole snapshot is
        # one immutable tuple replaced in a single assignment, so a reader can
        # never observe it half-built. Coordinates are full-frame pixels.
        #   (sequence, epoch, monotonic_time, (height, width), hands, markers)
        self.detections = None
        # Resolution of the display-only marker scan. 0 keeps it at full size.
        self.overlay_marker_width = config.get('preview', {}).get('marker_detect_width', 1280)

    def submit(self, frame, sequence, undistorter, profile):
        with self.lock:
            if self.active:
                self.latest = (frame, sequence, undistorter, profile, self.epoch)
        self.wake.set()

    def set_active(self, active):
        with self.lock:
            if self.active != active:
                self.active = active
                self.generation += 1
                self.epoch += 1
                self.latest = None
                self.detections = None
                if not active:
                    self.has_results = False
                    self.reset_requested = False
                    self.reset_ready = False
                    self.inspected_lines = set()
                    self.board = self._new_board()
        self.wake.set()

    def stop(self):
        with self.lock:
            self.stopped.set()
        self.wake.set()

    def poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            yield event

    def _status(self, text):
        if text != self.last_status:
            self.last_status = text
            self.events.put(('status', self.generation, text))

    def _reset_checkpoint(self):
        self.checkpoint = Checkpoint(self.settings['hand_absence_seconds'], 0.1,
                                     self.settings['min_hand_present_seconds'])
        self.presence = Debouncer(self.settings['debounce_frames'])

    def _complete(self):
        while True:
            try:
                job = self.prepared_jobs.get_nowait()
            except queue.Empty:
                break
            self.preparing_count -= 1
            self.gate_busy = self.preparing_count > 0
            if job is not None:
                if job['epoch'] != self.epoch or job['generation'] != self.generation or not self.active:
                    self._cancel_job(job, 'cancelled')
                    continue
                self.pending_jobs.append(job)
        if self.future is None and self.pending_jobs and self.active and not self.stopped.is_set():
            if (self.pending_jobs[0]['generation'] == self.generation
                    and self.pending_jobs[0]['epoch'] == self.epoch):
                self._start_job(self.pending_jobs.popleft())
            else:
                self._discard_pending('cancelled')
        self._check_reset_ready()
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
            if self.job_generation == self.generation and self.active and not self.stopped.is_set():
                if result is not None:
                    self.has_results = True
                    self.events.put(('result', self.generation, result))
            else:
                self._status('Previous capture saved; checking the latest fabric arrangement')
        except Exception as error:
            if self.job_generation == self.generation and self.active and not self.stopped.is_set():
                self.events.put(('error', self.generation, str(error)))
        if self.pending_jobs and self.active and not self.stopped.is_set():
            if (self.pending_jobs[0]['generation'] == self.generation
                    and self.pending_jobs[0]['epoch'] == self.epoch):
                self._start_job(self.pending_jobs.popleft())
            else:
                self._discard_pending('cancelled')
        self._check_reset_ready()

    @property
    def busy(self):
        return self.future is not None or bool(self.pending_jobs) or self.gate_busy

    def set_paused(self, paused):
        with self.lock:
            self.paused = bool(paused)
            self.pause_revision += 1
            self.latest = None
            self.detections = None
        self.wake.set()

    def finish_reset(self):
        # Called only after Tk displayed the last result for two seconds.
        with self.lock:
            self.has_results = False
            self.reset_requested = False
            self.reset_ready = False
            self.inspected_lines = set()
            self.board = self._new_board()
            self.epoch += 1
            self.latest = None
            self.detections = None
        self.wake.set()

    def _new_board(self):
        """A fresh record of the board's lines, or None outside scene mode.

        A new object rather than a cleared one: an inspection still running keeps
        the record it started with, so its late results cannot land on the next board.
        """
        if self.mode != 'scene':
            return None
        from inspection.board_lines import BoardLines
        return BoardLines(self.config.get('local_inspection', {}).get('board_match_distance_px', 60.0))

    def _hand_detect(self, hand, raw):
        """Return (present, landmarks) from whichever interface the hand offers.

        Tests inject a stand-in implementing only present(); it gets the raw
        frame and its verdict is used unchanged, exactly as before.
        """
        detect = getattr(hand, 'detect', None)
        if detect is None:
            return hand.present(raw), ()
        return detect(raw)

    def _publish_detections(self, sequence, epoch, now, size, hands, markers):
        self.detections = (sequence, epoch, now, size, hands, markers)

    def _check_reset_ready(self):
        if self.reset_requested and not self.busy and not self.reset_ready:
            self.reset_ready = True
            self.events.put(('reset_ready', self.generation, None))

    def _cancel_job(self, job, status):
        if job['path'] is not None:
            try:
                self.pipeline.store.finish(job['path'], {'status': status, 'capture': job['metadata']})
            except OSError as error:
                self.events.put(('error', job['generation'], str(error)))

    def _discard_pending(self, status):
        while self.pending_jobs:
            self._cancel_job(self.pending_jobs.popleft(), status)

    def _start_job(self, job):
        self._status('Inspecting captured fabric…')
        log.info('Inspection started for the capture of frame %s; lines already measured: %s; captures behind it: %d',
                 job['metadata'].get('frame_sequence'), sorted(self.inspected_lines),
                 self.preparing_count + len(self.pending_jobs))
        self.future = Future()
        self.job_generation = job['generation']
        # What this capture still owes, decided here rather than when it was
        # triggered: the lines nothing has measured yet, less one for each capture
        # already queued behind this one, which will measure them instead.
        with self.lock:
            job['metadata']['inspected_lines'] = sorted(self.inspected_lines)
            job['metadata']['queued_captures'] = self.preparing_count + len(self.pending_jobs)

        def process(future):
            try:
                if job['path'] is not None:
                    self.pipeline.store.update_metadata(job['path'], job['metadata'])
                def publish(result):
                    if job['generation'] == self.generation and not self.stopped.is_set():
                        self.has_results = True
                        with self.lock:
                            self.inspected_lines.add(result['measurement']['selected_line'])
                        self.events.put(('result', job['generation'], result))
                if self.mode == 'scene' and hasattr(self.pipeline, 'run_board'):
                    def shown(payload):
                        if job['generation'] == self.generation and not self.stopped.is_set():
                            # Even a board with no line yet is on the screen and is
                            # cleared with the bed, so it counts as a result.
                            self.has_results = True
                            with self.lock:
                                self.inspected_lines = {entry['line_id'] for entry in payload['lines']
                                                        if entry.get('measurement') is not None}
                            log.info('Frame %s, board: %s | lines %s | new from this photo %s | %.2f s',
                                     payload.get('frame_sequence'), payload['verdict'],
                                     [f"{entry['line_id']} {entry['status']}" for entry in payload['lines']],
                                     payload['new_lines'], payload['seconds'])
                            self.events.put(('board', job['generation'], payload))
                    self.pipeline.run_board(job['original'], job['corrected'], job['metadata'],
                        job['profile']['width'], job['profile']['tolerance'], job['path'],
                        board=job.get('board'), on_board=shown,
                        on_error=lambda message: self.events.put(('error', job['generation'], message)))
                    future.set_result(None)
                    return
                if hasattr(self.pipeline, 'run_batch'):
                    extra = {}
                    if 'on_missing' in inspect.signature(self.pipeline.run_batch).parameters:
                        # A line with no adhesive, when a coverage check is set, is a
                        # verdict (FAIL) rather than a processing error (WARNING).
                        extra['on_missing'] = lambda message: self.events.put(
                            ('line_failed', job['generation'], message))
                    self.pipeline.run_batch(job['original'], job['corrected'], job['metadata'],
                        job['profile']['width'], job['profile']['tolerance'], job['path'],
                        on_result=publish,
                        on_error=lambda message: self.events.put(('error', job['generation'], message)),
                        **extra)
                    future.set_result(None)
                    return
                future.set_result(self.pipeline.run(
                    job['original'], job['corrected'], job['metadata'],
                    job['profile']['width'], job['profile']['tolerance'], job['path'],
                    lambda message: self.events.put(('status', job['generation'], message))))
            except Exception as error:
                future.set_exception(error)
            finally:
                self.wake.set()
        threading.Thread(target=process, args=(self.future,),
                         name='automatic-inspection', daemon=True).start()

    @property
    def queue_text(self):
        return (f"Queue: {self.preparing_count} checking | "
                f"{len(self.pending_jobs)} waiting | "
                f"{int(self.future is not None)} inspecting")

    def _prepare_capture(self, gate, item):
        original, sequence, undistorter, profile, item_epoch, trigger_generation, metadata = item
        pipeline = self.pipeline
        capture_path = None
        try:
            if item_epoch != self.epoch or not self.active or self.stopped.is_set():
                return None
            corrected = None
            if self.config['capture_storage']['save_rejected_triggers']:
                corrected = undistorter.undistort(original) if undistorter else None
                capture_path = pipeline.store.begin(original, corrected, metadata)
            verdict = gate.classify(original)
            metadata['fabric'] = verdict
            if verdict['label'] != 'not_checked':
                log.info('Fabric check of frame %s: %s (%.0f %%), %s', sequence, verdict['label'],
                         100 * verdict['confidence'], 'accepted' if verdict['accepted'] else 'not accepted')
            if capture_path is not None:
                pipeline.store.update_metadata(capture_path, metadata)
            if item_epoch != self.epoch or not self.active or self.stopped.is_set():
                if capture_path is not None:
                    pipeline.store.finish(capture_path, {'status': 'cancelled', 'capture': metadata})
                return None
            if not verdict['accepted']:
                if capture_path is not None:
                    pipeline.store.finish(capture_path, {'status': 'rejected', 'capture': metadata})
                self._status(f"Fabric check: {verdict['label']} ({verdict['confidence']:.0%})")
                return None
            if undistorter is None:
                raise RuntimeError('Camera calibration is required before automatic measurement')
            if corrected is None:
                corrected = undistorter.undistort(original)
            job = {'original': original, 'corrected': corrected, 'metadata': metadata,
                   'profile': profile, 'path': capture_path, 'generation': trigger_generation,
                   'epoch': item_epoch, 'board': self.board}
            return job
        except Exception as error:
            if capture_path is not None:
                try:
                    pipeline.store.finish(capture_path, {'status': 'error', 'error': str(error)})
                except OSError:
                    pass
            self.events.put(('error', trigger_generation, str(error)))
            return None

    def _capture_worker(self, gate):
        while True:
            item = self.capture_jobs.get()
            if item is None:
                return
            job = self._prepare_capture(gate, item)
            with self.lock:
                if job is not None and self.stopped.is_set():
                    self._cancel_job(job, 'cancelled')
                else:
                    self.prepared_jobs.put(job)
            self.wake.set()

    def _queue_capture(self, raw, sequence, undistorter, profile, item_epoch, trigger):
        """Hand this frame to the fabric check and the inspection queue."""
        trigger_generation = self.generation
        # The hand result, fabric verdict, saved raw image and
        # undistortion all refer to this exact immutable snapshot.
        original = raw.copy()
        metadata = {'triggered_at': datetime.now().astimezone().isoformat(),
                    'frame_sequence': sequence, 'fabric': None,
                    'end_exclusion_percent': profile.get('end_exclusion_percent',
                        self.config['inspection'].get('end_exclusion_percent', 5.0)),
                    'size': profile['size'], 'target_width_mm': profile['width'],
                    'tolerance_mm': profile['tolerance']}
        if trigger != 'hand':
            metadata['trigger'] = trigger
        self.preparing_count += 1
        self.gate_busy = True
        self.capture_jobs.put((original, sequence, undistorter, dict(profile),
                               item_epoch, trigger_generation, metadata))

    SCENE_STATUS = {'BARE': 'Bed is empty — waiting for a board',
                    'MOVING': 'Movement — waiting for the board to be still',
                    'SETTLING': 'Still — confirming…',
                    'STILL': 'Board still — nothing new to measure',
                    'VETOED': 'Hand in the picture — waiting for it to leave'}

    def _scene_step(self, hand, raw, sequence, undistorter, profile, item_epoch, now, markers):
        """One frame of scene mode: a capture when the picture is still and has changed.

        No hand has to come and go. Movement, a hand's included, only delays the
        capture; the hand model is asked once about the still frame that would be
        captured, so that a hand resting in view is not measured as cloth.
        """
        small, motion = self._scene_signals(raw, self._scene_previous)
        self._scene_previous = small
        bare = len(markers) >= self.scene_settings.get('bare_bed_markers', 2)
        event = self.scene.update(motion, bare, small, now)
        hands = ()
        if event == 'capture' and hand is not None:
            present, hands = self._hand_detect(hand, raw)
            if present:
                self.scene.reject(now)
                log.info('Still frame %s not captured: a hand is in the picture', sequence)
                event = None
        self._publish_detections(sequence, item_epoch, now, raw.shape[:2], hands, markers)
        if self.scene.state != self._scene_logged:
            self._scene_logged = self.scene.state
            log.info('Scene %s at frame %s (%s)', self.scene.state, sequence, self.scene.describe())
        if event == 'cycle_end':
            log.info('Board left the bed at frame %s', sequence)
            if (self.has_results or self.busy) and not self.reset_requested:
                self.reset_requested = True
                self.events.put(('reset_wait', self.generation, 'Waiting for inspection results…'))
            else:
                self.board = self._new_board()
            return
        if event != 'capture':
            # A still board whose photo is still being read is not "nothing new".
            if self.scene.state == 'STILL' and self.busy:
                self._status('Measuring the board…')
            else:
                self._status(self.SCENE_STATUS.get(self.scene.state, self.scene.state))
            return
        self.scene.accept()
        self._status('Measuring the board…')
        log.info('Trigger: board still and changed; frame %s is captured', sequence)
        self._queue_capture(raw, sequence, undistorter, profile, item_epoch, 'scene')

    def _reset_scene(self):
        if self.scene is not None:
            self.scene.reset()
        self._scene_previous = None
        self._scene_logged = None

    def run(self):
        hand = None
        scene = self.mode == 'scene'
        try:
            if scene:
                from .scene_monitor import SceneMonitor, scene_signals
                self._scene_signals = scene_signals
                self.scene = SceneMonitor.from_settings(self.settings)
                self._reset_scene()
                self.board = self._new_board()
            self._status('Loading hand, fabric and local ONNX inspection models…')
            # In scene mode the hand model only vetoes a still frame, and can be left out.
            if not scene or self.scene_settings.get('hand_veto', True):
                hand = self.hand_factory(self.settings)
            if self.stopped.is_set():
                return
            if scene:
                gate = _NoGate()
            else:
                self._status('Loading fabric classification model…')
                gate = self.fabric_factory(self.settings)
            if self.stopped.is_set():
                return
            self._status('Loading adhesive inspection models…')
            self.pipeline = pipeline = self.pipeline_factory()
            if self.stopped.is_set():
                return
            threading.Thread(target=self._capture_worker, args=(gate,),
                             name='automatic-fabric-check', daemon=True).start()
            self._reset_checkpoint()
            last_seq, epoch = None, self.epoch
            pause_revision = self.pause_revision
            self.marker_monitor = BedMarkers(self.settings.get("marker_confirm_seconds", 0.5))
            self.ready = True
            self.events.put(('ready', self.generation, None))
            self._status('Automatic inspection ready — waiting for a board' if scene
                         else 'Automatic inspection ready — waiting for a hand')
            while not self.stopped.is_set():
                self._complete()
                self.wake.wait(0.05)
                self.wake.clear()
                with self.lock:
                    item, self.latest = self.latest, None
                    active, current_epoch = self.active, self.epoch
                if epoch != current_epoch:
                    self._discard_pending('cancelled')
                    epoch = current_epoch
                    last_seq = None
                    self._reset_checkpoint()
                    self.marker_monitor.reset()
                    self._reset_scene()
                if pause_revision != self.pause_revision:
                    pause_revision = self.pause_revision
                    self._reset_checkpoint()
                    self._reset_scene()
                if not active or item is None:
                    continue
                raw, sequence, undistorter, profile, item_epoch = item
                if item_epoch != epoch or sequence == last_seq:
                    continue
                last_seq = sequence
                capture_path = None
                try:
                    if item_epoch != self.epoch or not self.active:
                        continue
                    now = time.monotonic()
                    if scene:
                        pass        # the scene monitor ends the cycle, in _scene_step
                    elif self.has_results or self.busy:
                        if self.marker_monitor.update(raw, now) and not self.reset_requested:
                            self.reset_requested = True
                            self.events.put(('reset_wait', self.generation, 'Waiting for inspection results…'))
                    else:
                        self.marker_monitor.reset()
                    self._check_reset_ready()
                    # Preview geometry only. This is a second, downscaled scan
                    # that exists purely to be drawn; it cannot arm the reset
                    # above, which stays the full-resolution scan's decision.
                    markers = self.marker_monitor.detect_for_display(raw, self.overlay_marker_width)
                    if self.paused or self.reset_requested:
                        # No hand inference runs on this path, so publish the
                        # markers with no hands rather than leaving the last
                        # skeleton drawn over a frame that no longer has one.
                        self._publish_detections(sequence, item_epoch, now, raw.shape[:2], (), markers)
                        continue
                    if scene:
                        self._scene_step(hand, raw, sequence, undistorter, profile, item_epoch, now, markers)
                        continue
                    raw_present, hands = self._hand_detect(hand, raw)
                    if item_epoch != self.epoch or not self.active:
                        continue
                    self._publish_detections(sequence, item_epoch, now, raw.shape[:2], hands, markers)
                    present = self.presence.update(raw_present)
                    if present != getattr(self, '_logged_present', None):
                        self._logged_present = present
                        log.info('Hands %s at frame %s', 'in the picture' if present else 'out of the picture', sequence)
                    fired = self.checkpoint.update(present, time.monotonic())
                    if not fired:
                        if present:
                            self._status('Hand present — waiting for hands to leave')
                        elif self.checkpoint.state == Checkpoint.WAITING:
                            self._status('Hands clear — confirming absence…')
                        continue
                    self._status('Checking fabric…')
                    log.info('Trigger: hands left; frame %s is captured', sequence)
                    self._queue_capture(raw, sequence, undistorter, profile, item_epoch, 'hand')
                except Exception as error:
                    if capture_path is not None:
                        try:
                            pipeline.store.finish(capture_path, {'status': 'error', 'error': str(error)})
                        except OSError:
                            pass
                    self._reset_checkpoint()
                    self._reset_scene()
                    self.events.put(('error', self.generation, str(error)))
        except Exception as error:
            self.events.put(('error' if self.ready else 'startup_error', self.generation,
                             f'Automatic inspection unavailable: {error}'))
        finally:
            with self.lock:
                self.stopped.set()
                while True:
                    try:
                        job = self.prepared_jobs.get_nowait()
                    except queue.Empty:
                        break
                    if job is not None:
                        self._cancel_job(job, 'cancelled')
            self.capture_jobs.put(None)
            self._discard_pending('cancelled')
            if hand is not None:
                hand.close()
