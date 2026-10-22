"""Immutable capture -> local ONNX unfolding/refolding -> calibrated measurements."""
import logging
import time
import cv2
import numpy as np
import plane_scale
from .line_measurement import measurement_record, grading_checks


log = logging.getLogger('flod.inspection')

EDGE_REASON = 'line runs out of the photo — not fully in view'


def cut_by_edge(result, size, margin):
    """True when a measured line's glue reaches within ``margin`` px of the photo's border.

    ``size`` is the photo's (height, width); the mask lies at ``result['origin']``
    in it. A margin of 0 or less turns the check off.
    """
    if margin <= 0:
        return False
    ys, xs = np.nonzero(result['mask'])
    if not len(xs):
        return False
    left, top = result.get('origin', (0, 0))
    height, width = size
    return bool(xs.min() + left <= margin or ys.min() + top <= margin
                or xs.max() + left >= width - 1 - margin or ys.max() + top >= height - 1 - margin)


def describe(measurement):
    """One log line for a measured glue line."""
    def number(value, unit, digits):
        return 'none' if value is None else f'{value:.{digits}f} {unit}'
    ribbon = measurement.get('ribbon')
    # With end zones (inspection.end_exclusion_px) the two ends are named apart from the rest.
    segments = [s.get('average_width_mm') for s in measurement.get('segments', ()) if not s.get('end_zone')]
    zones = [s.get('average_width_mm') for s in measurement.get('segments', ()) if s.get('end_zone')]
    return (f"line {measurement['selected_line']} of {measurement['line_count']}: {measurement['status']}"
            f" | width {number(measurement.get('mean_width_px'), 'px', 1)}, {number(measurement.get('mean_width_mm'), 'mm', 2)}"
            f" | segments mm {['none' if s is None else round(s, 2) for s in segments]}"
            + (f" | end zones mm {['none' if s is None else round(s, 2) for s in zones]}" if zones else '')
            + f" | pieces {measurement['component_count']}, coverage {number(measurement.get('coverage_percent'), '%', 0)},"
            f" largest gap {number(measurement.get('largest_gap_mm'), 'mm', 1)}"
            + (f" | ribbon {ribbon['state']}" + (f" ({ribbon['reason']})" if ribbon.get('reason') else '')
               + f", mask width {number(ribbon.get('mask_width_px'), 'px', 1)}"
               + (f", ends moved {[round(v, 1) for v in ribbon['end_moved_px']]} px"
                  + ('' if all(ribbon.get('end_found', ())) else ' (an end was not found)')
                  if ribbon.get('end_moved_px') else '') if ribbon else '')
            + (f" | reasons: {'; '.join(measurement['fail_reasons'])}" if measurement.get('fail_reasons') else '')
            + f" | {number(measurement.get('line_seconds'), 's', 2)}")


class InspectionPipeline:
    def __init__(self, inspector, store, config):
        self.inspector, self.store, self.config = inspector, store, config

    def _ends(self):
        """End zones (inspection.end_exclusion_px, 0 or absent: off) for measurement_record."""
        settings = self.config['inspection']
        zone = settings.get('end_exclusion_px', 0)
        return dict(end_exclusion_px=float(zone) if zone else None,
                    end_trim_px=float(settings.get('end_trim_px', 0.0)))

    def _too_short(self, measurement):
        """Why a measured glue line is below the minimum length, or None.

        The measured glue itself is checked, in millimetres: a fabric partly
        outside the view, or a stray fragment, gives a short line that is not a
        whole glue line, so it is reported and not graded. Without millimetre
        calibration the length is unknown and nothing is rejected. Off unless
        inspection.min_line_length_mm is set. From the ``dev-lm`` branch.
        """
        minimum = self.config['inspection'].get('min_line_length_mm')
        length = measurement.get('length_mm')
        if not minimum or length is None or length >= minimum:
            return None
        return (f'glue line {length:.1f} mm is shorter than the {minimum:g} mm minimum; '
                'not graded. Place the whole fabric inside the camera view.')

    def run(self, original, undistorted, metadata, target_width, tolerance, capture_path=None,
            progress=lambda text: None):
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_rightmost_line'}
        try:
            result = self.inspector.run(undistorted, progress)
            warnings = []
            height, width = undistorted.shape[:2]
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)),
                    checks=grading_checks(self.config['inspection']), **self._ends())
            if measurement['status'] == 'WARNING':
                warnings.append('Pixel measurements available; calibrated widths incomplete or unavailable')
            self.store.save_mask(path, result['mask'])
            self.store.save_overlay(path, result['overlay'])
            self.store.save_preview(path, 'overview', result['overlay'])
            record.update(status='complete', frame_size=[width, height], warnings=warnings,
                          settings=self.config['local_inspection'], measurement=measurement)
            return {'measurement': measurement, 'overlay': result['overlay'], 'mask': result['mask'],
                    'source_image': undistorted,
                    'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)

    def run_batch(self, original, undistorted, metadata, target_width, tolerance,
                  capture_path=None, on_result=lambda result: None, on_error=lambda message: None,
                  on_missing=None):
        """``on_missing(message)`` is told about a line with no measurable adhesive
        when a coverage check is set; without it, or without the check, such a
        line is reported through ``on_error`` as before."""
        from .overlay import annotated_overlay
        from .local.engine import NoAdhesiveError
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_lines', 'lines': []}
        started = time.perf_counter()
        try:
            height, width = undistorted.shape[:2]
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
                log.warning('Millimetre calibration unavailable: %s', error)
            checks = grading_checks(self.config['inspection'])
            failures = []
            def failed_line(index, error):
                message = f'Line {index}: {error}'
                failures.append(message)
                if (on_missing is not None and isinstance(error, NoAdhesiveError)
                        and checks['min_coverage_percent'] > 0):
                    # No adhesive at all is the extreme case of too little of it.
                    record.setdefault('missing_lines', []).append(index)
                    on_missing(f'Line {index}: no adhesive found (FAIL)')
                else:
                    on_error(message)
            for result in self.inspector.run_lines(
                    undistorted, inspected=metadata.get('inspected_lines', ()),
                    reserved=metadata.get('queued_captures', 0), on_error=failed_line):
                measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)),
                    checks=checks, **self._ends())
                short = self._too_short(measurement)
                if short:
                    # Not published or marked measured: the line is offered again
                    # on the next trigger, once the whole fabric is in view.
                    log.warning('Frame %s, line %s: %s', metadata.get('frame_sequence'),
                                result['selected_line'], short)
                    failed_line(result['selected_line'], short)
                    continue
                # The mask is what the operator sees as the strip, so it is
                # drawn under the graded centrelines rather than instead of them.
                overlay = annotated_overlay(undistorted, measurement, result['mask'])
                line_path = path / f"line_{result['selected_line']}"
                line_path.mkdir()
                self.store.save_mask(line_path, result['mask'])
                self.store.save_overlay(line_path, overlay)
                self.store.save_preview(line_path, 'overview', overlay)
                self.store.finish(line_path, {'status': 'complete', 'measurement': measurement})
                record['lines'].append(measurement)
                log.info('Frame %s, %s', metadata.get('frame_sequence'), describe(measurement))
                line_warnings = list(warnings)
                if measurement['status'] == 'WARNING':
                    line_warnings.append('Calibrated widths incomplete or unavailable')
                on_result({'measurement': measurement, 'overlay': overlay, 'mask': result['mask'],
                           'source_image': undistorted,
                           'warnings': line_warnings, 'path': str(line_path)})
            record.update(status='complete_with_errors' if failures else 'complete',
                          errors=failures, warnings=warnings, frame_size=[width, height])
            log.info('Frame %s inspected: %d line(s) measured, %d failed, %.2f s, saved in %s',
                     metadata.get('frame_sequence'), len(record['lines']), len(failures),
                     time.perf_counter() - started, path)
        except Exception as error:
            record['error'] = str(error)
            log.exception('Inspection of frame %s failed', metadata.get('frame_sequence'))
            raise
        finally:
            self.store.finish(path, record)

    def run_board(self, original, undistorted, metadata, target_width, tolerance,
                  capture_path=None, board=None, on_board=lambda payload: None,
                  on_error=lambda message: None):
        """Scene mode: one photo of the whole board; only lines not read before are measured.

        ``board`` (a ``BoardLines``) remembers the lines of the board in front of
        the camera from photo to photo and keeps the first good reading of each;
        without one, every line of the photo is measured. ``on_board(payload)`` is
        called exactly once, with every line known so far; a photo with no source
        line at all (a bare board) is a normal outcome, not an error.
        ``on_error(message)`` is told about each line that could not be read; such
        a line is tried again on the next photo.

        What is saved: ``board.jpg`` and ``result.json`` in the set, and a folder
        ``line_<id>`` with the mask cut round the line and its ``result.json`` for
        each line read from this photo. Everything is written after ``on_board``,
        the two lossless source pictures last, so the operator is not kept waiting
        for the disk.
        """
        from .board_lines import BoardLines, course
        from .board_overlay import board_overview, aggregate_verdict
        from .local.engine import NoAdhesiveError, NoSourceLinesError
        started = time.perf_counter()
        deferred = capture_path is None
        path = capture_path or self.store.begin_deferred(metadata)
        if not deferred:
            self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_board', 'lines': []}
        board = BoardLines() if board is None else board
        sequence = metadata.get('frame_sequence')
        try:
            height, width = undistorted.shape[:2]
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
                log.warning('Millimetre calibration unavailable: %s', error)
            checks = grading_checks(self.config['inspection'])
            edge_margin = self.config.get('local_inspection', {}).get('board_edge_margin_px', 20)
            try:
                source, lines = self.inspector.find_lines(undistorted, sideways=True)
            except NoSourceLinesError:
                source, lines = None, []        # a bare board
            found_at = time.perf_counter()
            picks = board.owed(lines)
            wanted = {line_id: lines[index] for index, line_id in picks}
            failures, new_lines = [], []

            def failed_line(line_id, error):
                message = f'Line {line_id}: {error}'
                failures.append(message)
                missing = isinstance(error, NoAdhesiveError) and checks['min_coverage_percent'] > 0
                board.fail(line_id, {
                    'line_id': line_id, 'status': 'FAIL', 'measurement': None,
                    'error': 'no adhesive found' if missing else str(error),
                    'centre': course(wanted[line_id]['original_xy']), 'outline': None,
                    'crop': None, 'crop_mask': None, 'origin': (0, 0),
                    'new': True, 'photo_sequence': sequence})
                log.warning('Frame %s, %s', sequence, message)
                on_error(message)

            for result in self.inspector.measure_found(
                    undistorted, source, [(lines[index], line_id) for index, line_id in picks],
                    board.count, on_error=failed_line):
                line_id = result['selected_line']
                measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)),
                    checks=checks, **self._ends())
                short = self._too_short(measurement)
                if short:
                    failed_line(line_id, short)
                    continue
                if measurement['status'] == 'FAIL' and cut_by_edge(result, (height, width), edge_margin):
                    # Part of the line lies outside the photo, so coverage and the
                    # end beside the edge read short: it cannot be graded, as an
                    # ungradeable segment is WARNING (line_measurement._grade).
                    measurement['cut_by_edge'] = True
                    measurement['status'] = 'WARNING'
                    measurement['fail_reasons'] = [EDGE_REASON] + list(measurement.get('fail_reasons') or [])
                board.record(line_id, self._board_entry(undistorted, result, measurement, sequence))
                new_lines.append(line_id)
                record['lines'].append(measurement)
                log.info('Frame %s, %s', sequence, describe(measurement))
            measured_at = time.perf_counter()
            entries = board.entries
            overview = board_overview(undistorted, entries, 1920)
            seconds = time.perf_counter() - started
            on_board({'lines': entries, 'overview': overview, 'verdict': aggregate_verdict(entries),
                      'new_lines': list(new_lines), 'seconds': seconds, 'path': str(path),
                      'frame_sequence': sequence, 'source_image': undistorted})
            shown_at = time.perf_counter()
            for entry in entries:
                if entry['line_id'] in new_lines:
                    line_path = path / f"line_{entry['line_id']}"
                    line_path.mkdir(exist_ok=True)
                    self.store.save_mask(line_path, entry['crop_mask'])
                    self.store.finish(line_path, {'status': 'complete', 'line_id': entry['line_id'],
                                                  'origin': list(entry['origin']),
                                                  'measurement': entry['measurement']})
            self.store.save_preview(path, 'board', overview)
            if deferred:
                self.store.save_sources(path, original, undistorted)
            record.update(status='complete_with_errors' if failures else 'complete',
                          known_lines=[entry['line_id'] for entry in entries], new_lines=list(new_lines),
                          errors=failures, warnings=warnings, frame_size=[width, height],
                          verdict=aggregate_verdict(entries), seconds=seconds)
            log.info('Frame %s board: %d line(s) found, %d read now (%s), %d failed, %d known; '
                     'finding %.2f s, measuring %.2f s, picture %.2f s, shown after %.2f s, saving %.2f s, in %s',
                     sequence, len(lines), len(new_lines), ', '.join(map(str, new_lines)) or 'none',
                     len(failures), len(entries), found_at - started, measured_at - found_at,
                     shown_at - measured_at, seconds, time.perf_counter() - shown_at, path)
        except Exception as error:
            record['error'] = str(error)
            log.exception('Board inspection of frame %s failed', sequence)
            raise
        finally:
            self.store.finish(path, record)

    @staticmethod
    def _board_entry(photo, result, measurement, sequence):
        """One line of the board as the board view shows it; see the payload in run_board.

        The photo and the mask are cut round the line, with the room the single-line
        view leaves (``result_view.show_inspection``): 160 and 180 px to the sides
        of the line, 40 px beyond its ends. A copy is cut, so that keeping a line's
        reading does not keep the whole photo it came from.
        """
        mask = result['mask']
        left, top = result.get('origin', (0, 0))    # where the mask lies in the photo
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = int(xs.min()) + left, int(xs.max()) + left, int(ys.min()) + top, int(ys.max()) + top
        upright = y1 - y0 >= x1 - x0
        (before_x, after_x), (before_y, after_y) = ((160, 180), (40, 40)) if upright else ((40, 40), (160, 180))
        height, width = photo.shape[:2]
        x0, x1 = max(0, x0 - before_x), min(width, x1 + after_x)
        y0, y1 = max(0, y0 - before_y), min(height, y1 + after_y)
        crop_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        # The part of the mask that lies inside the cut, put at its place in it.
        mx0, my0 = max(x0, left), max(y0, top)
        mx1, my1 = min(x1, left + mask.shape[1]), min(y1, top + mask.shape[0])
        crop_mask[my0-y0:my1-y0, mx0-x0:mx1-x0] = mask[my0-top:my1-top, mx0-left:mx1-left]
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = (max(contours, key=cv2.contourArea)[:, 0, :].astype(float) + (x0, y0)) if contours else None
        main = max(measurement['components'], key=lambda component: component['length_px'])
        return {'line_id': result['selected_line'], 'status': measurement['status'],
                'measurement': measurement, 'error': None,
                'centre': np.array([[s['center_x'], s['center_y']] for s in main['samples']], dtype=float),
                'outline': outline, 'crop': photo[y0:y1, x0:x1].copy(), 'crop_mask': crop_mask,
                'origin': (x0, y0), 'new': True, 'photo_sequence': sequence}
