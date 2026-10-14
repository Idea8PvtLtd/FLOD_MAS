"""Tk integration; cameras and calibration remain owned by the existing app."""
import tkinter as tk
from tkinter import ttk
from ui_theme import COLORS as C, FONT
import time
from app_config import CONFIG, project_path
from auto_trigger.controller import AutoController
from pathlib import Path
from .local.engine import LiveInspection
from .pipeline import InspectionPipeline
from .storage import CaptureStore
from .result_view import LineResultView
from .board_view import BoardResultView
import logging

log = logging.getLogger('flod.dashboard')

# Detections older than this are not drawn. The worker cannot publish once the
# feed is paused, switched off or the thread has failed, so this age is what
# clears the overlay in those cases instead of leaving stale boxes on screen.
OVERLAY_MAX_AGE = 0.5


def _open_result_view(app, view_class):
    """Swap the camera view for a result view and move the progress widgets beside it.

    Shared by the line-by-line results and the board results of the scene mode,
    so both are laid out the same way and ``_dismiss_result_view`` undoes either.
    """
    # Fill the physical screen, retaining the existing application chrome;
    # result_view.fill_screen false keeps the window at its own size.
    if CONFIG.get('result_view', {}).get('fill_screen', True):
        app.root.geometry(f'{app.root.winfo_screenwidth()}x{app.root.winfo_screenheight()}+0+0')
    app.bottom_panel.place_forget()
    app.images_container.pack_forget()
    app.result_view = view_class(
        app.image_panel,
        show_live_preview=CONFIG.get('result_view', {}).get('show_live_preview', True))
    app.result_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=(58, 8))
    app.result_view.on_grade_changed = app.set_pass_fail
    app.result_view.set_paused(app.auto_controller.paused)
    app.action_overlay.place_configure(y=8)
    if app.result_view.show_live_preview:
        app.loading_container.place(in_=app.result_view.live_panel, x=0, rely=1,
                                    y=-12, anchor='sw', width=240)
        app.progress_label.configure(wraplength=212)
        app.inspection_queue_label.configure(font=(FONT, 8), padx=4, wraplength=232)
        app.inspection_queue_label.place_configure(height=38)
        app.progress_label.pack_configure(pady=(40, 2))
    app.action_overlay.lift()
    app.loading_container.lift()
    app.inspection_queue_label.lift()


class AutomaticDashboard:
    def _init_automatic(self):
        self._models_loading = True
        self._startup_failed = False
        self._show_startup_loading()
        def factory():
            store = CaptureStore(project_path(CONFIG['capture_storage']['directory']),
                                 CONFIG['capture_storage'])
            settings = CONFIG['local_inspection']
            root = Path(__file__).resolve().parents[1]
            inspector = LiveInspection(root / settings['source_model'], root / settings['glue_model'],
                                       settings['offset_pixels'], settings['source_width'],
                                       smooth_normal_rows=settings.get('smooth_normal_rows', 0),
                                       smooth_logits=settings.get('smooth_logits', False),
                                       min_line_length_ratio=settings.get('min_line_length_ratio', 0.),
                                       ribbon_finish=settings.get('ribbon_finish', False),
                                       segments=CONFIG['inspection'].get('segment_count', 10),
                                       execution_provider=settings.get('execution_provider', 'cpu'),
                                       line_workers=settings.get('line_workers', 1))
            return InspectionPipeline(inspector, store, CONFIG)
        self.auto_controller = AutoController(CONFIG, factory)
        self._reset_due = None
        self.auto_controller.start()
        self._auto_poll_job = self.root.after(100, self._poll_automatic)

    def _show_startup_loading(self):
        self.startup_panel = tk.Frame(self.image_panel, bg=C['camera'])
        self.startup_panel.place(x=0, y=0, relwidth=1, relheight=1)
        card = tk.Frame(self.startup_panel, bg=C['surface'], padx=40, pady=30)
        card.place(relx=.5, rely=.5, anchor='center')
        tk.Label(card, text='PREPARING INSPECTION', bg=C['surface'], fg=C['text'],
                 font=(FONT, 20, 'bold')).pack(pady=(0, 12))
        self.startup_description = tk.Label(card, text='Loading models…', bg=C['surface'],
            fg=C['muted'], font=(FONT, 11), wraplength=520, justify='center')
        self.startup_description.pack(pady=(0, 18))
        self.startup_bar = ttk.Progressbar(card, mode='indeterminate', length=420,
                                         style='App.Horizontal.TProgressbar')
        self.startup_bar.pack(fill='x')
        self.startup_bar.start(25)
        self.startup_panel.lift()
        self.action_overlay.lift()

    def _hide_startup_loading(self):
        if hasattr(self, 'startup_panel'):
            self.startup_bar.stop()
            self.startup_panel.place_forget()
        for name in ('bottom_panel', 'loading_container', 'action_overlay', 'inspection_queue_label'):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.lift()

    def _set_activity_animation(self, busy):
        if getattr(self, '_activity_animating', False) == bool(busy):
            if not busy:
                self.progress_bar.configure(value=0)
            return
        self._activity_animating = bool(busy)
        self.progress_bar.stop()
        if busy:
            self.progress_bar.configure(mode='indeterminate')
            self.progress_bar.start(35)
        else:
            self.progress_bar.configure(mode='determinate', value=0)

    def _poll_automatic(self):
        if getattr(self, '_closing', False):
            return
        controller = self.auto_controller
        active = (not self.video_paused and self.video_streaming
                  and getattr(self, 'calibration_page', None) is None)
        controller.set_active(active)
        if not active:
            self._reset_due = None
        for event, generation, payload in controller.poll():
            if event == 'ready':
                self._models_loading = False
                if getattr(self, 'calibration_page', None) is None and not self.video_paused:
                    self.startup_description.configure(text='Models ready. Opening camera…')
                    self.start_video_stream()
            elif event == 'startup_error':
                self._startup_failed = True
                self.startup_bar.stop()
                self.startup_description.configure(text=payload, fg=C['danger'])
                self.set_pass_fail('MODEL ERROR')
            elif event == 'result':
                if generation != controller.generation or not active:
                    continue
                self._show_automatic_results(payload)
            elif event == 'board':
                # Scene mode only: one payload holds every line of the board so far.
                if generation != controller.generation or not active:
                    continue
                self._show_board_results(payload)
            elif event == 'reset_wait':
                if active and generation == controller.generation:
                    self.update_progress(100, payload)
            elif event == 'reset_ready':
                if active and generation == controller.generation:
                    self._reset_due = time.monotonic() + 2.0
            elif event == 'clear':
                if active and generation == controller.generation:
                    self._dismiss_result_view()
                    self.set_pass_fail('READY')
                    self.update_progress(100, payload)
            elif event == 'error':
                if not active or generation != controller.generation:
                    continue
                self.set_pass_fail('WARNING')
                self.update_progress(100, payload)
            elif event == 'line_failed':
                if not active or generation != controller.generation:
                    continue
                self.set_pass_fail('FAIL')
                self.update_progress(100, payload)
            elif event == 'status':
                if getattr(self, '_models_loading', False) or (active and generation == controller.generation):
                    self.update_progress(100, payload)
        queue_label = getattr(self, 'inspection_queue_label', None)
        if queue_label is not None:
            queue_label.configure(text=controller.queue_text)
        if hasattr(self, '_set_activity_animation'):
            self._set_activity_animation(controller.busy and active)
            if controller.busy and active:
                text = ('Inspecting captured fabric…' if controller.future is not None
                        else 'Checking queued captures…' if controller.gate_busy
                        else 'Waiting for inspection…')
                self.progress_label.configure(text=text)
        if controller.reset_requested and not controller.paused:
            self.update_progress(100, 'Waiting for inspection results…' if self._reset_due is None
                                 else 'Results complete — returning to live…')
        if (self._reset_due is not None and not controller.paused
                and time.monotonic() >= self._reset_due):
            self._dismiss_result_view()
            self._reset_due = None
            controller.finish_reset()
            self.set_pass_fail('READY')
            self.update_progress(100, 'Paused' if controller.paused else 'Ready — waiting for a hand cycle')
        self._auto_poll_job = self.root.after(100, self._poll_automatic)

    def _toggle_automatic_pause(self):
        controller = getattr(self, 'auto_controller', None)
        if controller is None:
            return
        controller.set_paused(not controller.paused)
        self.auto_pause_button.configure(text='PLAY' if controller.paused else 'PAUSE')
        if getattr(self, 'result_view', None) is not None:
            self.result_view.set_paused(controller.paused)
        self.update_progress(100, 'Automatic capture paused · wheel to zoom, drag to pan'
                             if controller.paused else 'Automatic capture running')

    def _overlay_geometry(self):
        """Latest hand and marker geometry for the preview, or None to draw nothing.

        Read on the Tk thread only. None means "nothing to show", whether that
        is because the overlays are switched off, no detection has happened, the
        camera session has changed, or the last detection has gone stale.
        """
        if not getattr(self, 'show_overlays', False):
            return None
        controller = getattr(self, 'auto_controller', None)
        snapshot = getattr(controller, 'detections', None)
        if snapshot is None:
            return None
        _, epoch, stamp, size, hands, markers = snapshot
        if epoch != controller.epoch or time.monotonic() - stamp > OVERLAY_MAX_AGE:
            return None
        if not hands and not markers:
            return None
        return {'hands': hands, 'markers': markers, 'size': size}

    def _submit_automatic_frame(self, raw, sequence):
        if getattr(self, 'auto_controller', None) is not None:
            self.auto_controller.submit(raw, sequence, self.camera1.undistorter,
                {'size': self.active_size, 'width': self.active_strip_width,
                 'tolerance': self.active_strip_width_tolerance,
                 'end_exclusion_percent': self.active_end_exclusion_percent})

    def _show_automatic_results(self, result):
        measurement = result['measurement']
        if measurement and getattr(self, 'result_view', None) is None:
            _open_result_view(self, LineResultView)
        if measurement:
            self.result_view.show_inspection(result)
        status = 'WARNING' if result['warnings'] else measurement['status']
        log.info('Shown to the operator: line %s as %s%s', measurement['selected_line'], status,
                 f" (warnings: {' | '.join(result['warnings'])})" if result['warnings'] else '')
        self.set_pass_fail(status)
        # Why a line failed an extra check (a short defect, missing adhesive) is
        # not visible from the segment averages, so it is said in words.
        reasons = measurement.get('fail_reasons') or []
        self.update_progress(100, ' | '.join(result['warnings']) or
                             (f"Line #{measurement['selected_line']}: " + ' | '.join(reasons) if reasons else
                              f"Line #{measurement['selected_line']} of {measurement['line_count']} "
                              'measured — results saved — monitoring hands'))

    def _show_board_results(self, payload):
        """Scene mode: show every line of the board on one screen, from one payload."""
        view = getattr(self, 'result_view', None)
        if view is not None and not isinstance(view, BoardResultView):
            # A line-by-line view left over from the other mode cannot draw a board.
            self._dismiss_result_view()
            view = None
        if view is None:
            _open_result_view(self, BoardResultView)
        self.result_view.show_board(payload)
        verdict = payload['verdict']
        lines = payload['lines']
        new = payload.get('new_lines') or []
        log.info('Shown to the operator: board with %s line(s) as %s (new: %s)', len(lines), verdict,
                 ', '.join(str(line) for line in new) or 'none')
        # A board with no glue line yet has no verdict; the label stays neutral.
        self.set_pass_fail('READY' if verdict == 'EMPTY' else verdict)
        failed = [str(entry['line_id']) for entry in lines
                  if entry.get('measurement') is None or entry.get('status') == 'FAIL']
        self.update_progress(100, 'Board on the bed — no glue line yet' if not lines else
                             f"Board: {len(lines)} line{'' if len(lines) == 1 else 's'} — "
                             + (f"line {', '.join(failed)} failed" if failed else verdict)
                             + ' — results saved')

    def _dismiss_result_view(self):
        if getattr(self, 'result_view', None) is not None:
            self.result_view.destroy()
            self.result_view = None
            self.images_container.pack(**self.images_container_pack_opts)
            self.loading_container.place(in_=self.image_panel, x=20, rely=1, y=-20,
                                         anchor='sw', width=520)
            self.progress_label.configure(wraplength=490)
            self.progress_label.pack_configure(pady=(24, 2))
            self.inspection_queue_label.configure(font=(FONT, 10), padx=16, wraplength=0)
            self.inspection_queue_label.place_configure(height=22)
            self.bottom_panel.place(x=20, y=20)
            self.action_overlay.place_configure(y=20)
            self.bottom_panel.lift()
            self.loading_container.lift()
            self.action_overlay.lift()
            self.inspection_queue_label.lift()

    def _refresh_inspection_availability(self):
        if self.camera1 is None or not self.camera1.calibration_available:
            log.warning('Camera calibration is missing: automatic measurement cannot run')
            self.set_pass_fail('SETUP')
            self.update_progress(100, 'Camera calibration required; open Settings → Camera setup')
        else:
            self.set_pass_fail('READY')
            self.update_progress(100, 'Automatic inspection — waiting for a hand cycle')

    def reset_dashboard(self):
        self.auto_controller.set_active(False)
        self._dismiss_result_view()
        self.on_reset_callback()
