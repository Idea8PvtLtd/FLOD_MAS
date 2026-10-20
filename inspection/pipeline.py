"""Immutable capture -> local ONNX unfolding/refolding -> calibrated measurements."""
import plane_scale
from .line_measurement import measurement_record, rounded_record


class InspectionPipeline:
    def __init__(self, inspector, store, config):
        self.inspector, self.store, self.config = inspector, store, config

    def _too_short(self, measurement):
        """Why a measured glue line is below the minimum length, or None.

        The measured glue itself is checked, in millimetres: a fabric partly
        outside the view, or a stray fragment, gives a short line that is not a
        whole glue line, so it is reported and not graded. Without millimetre
        calibration the length is unknown and nothing is rejected.
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
                    self.config['inspection'].get('end_exclusion_px'),
                    self.config['inspection'].get('end_trim_px', 0.0),
                    self.config['inspection'].get('end_zone_min_width_mm'))
            if measurement['status'] == 'WARNING':
                warnings.append('Pixel measurements available; calibrated widths incomplete or unavailable')
            self.store.save_mask(path, result['mask'])
            self.store.save_overlay(path, result['overlay'])
            self.store.save_preview(path, 'overview', result['overlay'])
            record.update(status='complete', frame_size=[width, height], warnings=warnings,
                          settings=self.config['local_inspection'], measurement=rounded_record(measurement))
            return {'measurement': measurement, 'overlay': result['overlay'], 'mask': result['mask'],
                    'source_image': undistorted,
                    'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)

    def run_batch(self, original, undistorted, metadata, target_width, tolerance,
                  capture_path=None, on_result=lambda result: None, on_error=lambda message: None):
        from .overlay import annotated_overlay
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_lines', 'lines': []}
        try:
            height, width = undistorted.shape[:2]
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            failures = []
            def failed_line(index, error):
                message = f'Line {index}: {error}'
                failures.append(message)
                on_error(message)
            for result in self.inspector.run_lines(
                    undistorted, inspected=metadata.get('inspected_lines', ()),
                    reserved=metadata.get('queued_captures', 0), on_error=failed_line,
                    measured=metadata.get('measured_lines', ())):
                measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)),
                    self.config['inspection'].get('end_exclusion_px'),
                    self.config['inspection'].get('end_trim_px', 0.0),
                    self.config['inspection'].get('end_zone_min_width_mm'))
                short = self._too_short(measurement)
                if short:
                    # Not published or marked measured: the line is offered again
                    # on the next trigger, once the whole fabric is in view.
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
                saved = rounded_record(measurement)   # only final 0.1 mm values are kept
                self.store.finish(line_path, {'status': 'complete', 'measurement': saved})
                record['lines'].append(saved)
                line_warnings = list(warnings)
                if measurement['status'] == 'WARNING':
                    line_warnings.append('Calibrated widths incomplete or unavailable')
                on_result({'measurement': measurement, 'overlay': overlay, 'mask': result['mask'],
                           'source_image': undistorted, 'fingerprint': result.get('fingerprint'),
                           'warnings': line_warnings, 'path': str(line_path)})
            record.update(status='complete_with_errors' if failures else 'complete',
                          errors=failures, warnings=warnings, frame_size=[width, height])
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)
