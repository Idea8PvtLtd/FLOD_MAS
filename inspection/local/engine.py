"""In-memory source-line inspection with shared source detection per capture."""
from time import perf_counter
import numpy as np
from .geometry import (cv2, extract_lines, filter_wave_lines, offset_lines,
    region_mask, original_dimensions, unfold_maps, enhance_image, clean_mask,
    individual_crop, mapped_mask, smoothed_path)
from .width_measurement import measure_component


class NoAdhesiveError(ValueError):
    """The band of a line holds no adhesive that can be measured."""


class LiveInspection:
    def __init__(self, source_path, glue_path, offset=100., source_width=1152, smooth_normal_rows=0,
                 smooth_logits=False, min_line_length_ratio=0.):
        from .source_model import SourceMaskOnnx
        from .glue_model import SegformerOnnx
        self.source = SourceMaskOnnx(source_path, width=source_width, flip=False)
        self.glue = SegformerOnnx(glue_path, smooth_logits=smooth_logits)
        self.offset = float(offset)
        if not np.isfinite(self.offset) or self.offset <= 0:
            raise ValueError('Inspection offset must be positive and finite')
        # 0 keeps every source line. Above 0, a line shorter than this share of the
        # longest one in the capture is dropped before lines are numbered: a soft
        # picture can give short false edges, and every false line shifts the
        # left-to-right positions that later captures rely on.
        self.min_line_length_ratio = float(min_line_length_ratio)
        if not np.isfinite(self.min_line_length_ratio) or not 0 <= self.min_line_length_ratio < 1:
            raise ValueError('Minimum line length ratio must be at least 0 and less than 1')
        # 0 keeps the band as it always was; see geometry.offset_path.
        self.smooth_normal_rows = int(smooth_normal_rows)
        if self.smooth_normal_rows < 0:
            raise ValueError('Normal smoothing must be zero or a positive number of rows')

    def run(self, photo, progress=lambda text: None):
        """Single-line callers want the rightmost line and nothing else."""
        return next(self.run_lines(photo, progress, only_rightmost=True))

    def run_lines(self, photo, progress=lambda text: None, inspected=(), reserved=0,
                  only_rightmost=False, on_error=None):
        """Measure the lines this capture still owes, leftmost first.

        Positions are counted from the left and name the same physical line in
        every capture of a cycle, because lines are only ever added on the right.
        `inspected` holds the positions already measured, so a line that failed
        and was never measured is offered again rather than skipped over.
        `reserved` is how many captures are already queued behind this one: each
        will measure one more of the lines still outstanding, so this capture
        leaves that many for them instead of measuring the same line twice.
        """
        started = perf_counter()
        progress('AI source mask')
        source = self.source.predict(photo)
        chains = extract_lines(source, hook_zone=.04, hook_angle=65)
        waves = np.zeros_like(source)
        for chain in chains:
            cv2.polylines(waves, [chain], False, 255, 1, cv2.LINE_8)
        filtered, _, _ = filter_wave_lines(waves, source, .008, .1, .9, make_preview=False)
        _, _, lines = offset_lines(filtered, self.offset, 'inside', source_mask=source, make_preview=False,
                                   smooth_rows=getattr(self, 'smooth_normal_rows', 0))
        if not lines:
            raise ValueError('No accepted source lines. Check placement or the source mask model.')
        ratio = getattr(self, 'min_line_length_ratio', 0.)
        if ratio > 0 and len(lines) > 1:
            lengths = [float(np.ptp(np.asarray(line['original_xy'], dtype=float), axis=0).max()) for line in lines]
            lines = [line for line, length in zip(lines, lengths) if length >= ratio * max(lengths)]
        lines.sort(key=lambda line: float(np.mean(np.asarray(line['original_xy'])[:, 0])))
        total = len(lines)
        if only_rightmost:
            indices = [total - 1]
        else:
            measured = {int(position) for position in inspected if 1 <= int(position) <= total}
            outstanding = [position for position in range(1, total + 1)
                           if position not in measured]
            # The rightmost of what is outstanding is what this capture owes once
            # the queued captures have taken their share of it.
            indices = [position - 1 for position in outstanding[max(0, int(reserved)):]]
        try:
            for index in indices:
                progress(f'Inspecting line {index+1} of {len(lines)}')
                try:
                    yield self._inspect_line(photo, source, lines[index], index+1, len(lines), started, progress)
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(index+1, error)
        finally:
            # The enhanced photo is kept only while this capture's lines are measured.
            self._enhanced = None

    def _enhanced_photo(self, photo):
        """The contrast-enhanced photo, computed once per capture.

        Every line of a capture is cut from the same enhanced photo. Enhancing the
        whole 16-megapixel frame again for each line gave the same picture each
        time and was the largest single cost of a capture.
        """
        cached = getattr(self, '_enhanced', None)
        if cached is None or cached[0] is not photo:
            cached = (photo, enhance_image(photo, 2., 16, 16))
            self._enhanced = cached
        return cached[1]

    def _inspect_line(self, photo, source, line, index, line_count, started, progress):
        line_started = perf_counter()
        selected = region_mask(source.shape, [line])
        width, height = original_dimensions(line, self.offset)
        mx, my, _ = unfold_maps(line, width, height, 'horizontal')
        progress('Enhancing and unfolding the selected inward band')
        flat = cv2.remap(self._enhanced_photo(photo), mx, my, cv2.INTER_LINEAR)
        allowed = cv2.remap(selected, mx, my, cv2.INTER_NEAREST)
        flat[allowed == 0] = 0
        progress('Local SegFormer adhesive detection')
        predicted = self.glue.predict(flat)
        scale = width / self.glue.width
        predicted = clean_mask(predicted, max(1, round(9*scale)), max(0, round(12*scale)), .15)
        predicted[allowed == 0] = 0
        progress('Cleaning and refolding adhesive into the undistorted frame')
        _, crop_mask, bbox = individual_crop(photo, line)
        local, _, _, _ = mapped_mask(predicted, mx, my, bbox, crop_mask)
        mask = np.zeros_like(source)
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = local
        mask[selected == 0] = 0
        if not mask.any():
            raise NoAdhesiveError(f'Line {index}: adhesive model returned an empty mask.')
        progress('Measuring strip in undistorted frame pixels')
        # Measure each real component. Do not bridge disconnected glue across long gaps.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        measurements = []
        overlay = photo.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        for i in range(1, count):
            cx, cy, cw, ch, area = stats[i]
            if area < 100:
                continue
            component = np.uint8(labels[cy:cy+ch, cx:cx+cw] == i)*255
            component = np.pad(component, 2)
            try:
                result = measure_component(component)
            except ValueError:
                continue
            shift = np.array([cx-2, cy-2])
            points = np.array([[s['center_x'], s['center_y']] for s in result['samples']]) + shift
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], False, (0, 255, 255), 2)
            for sample in result['samples']:
                for key, delta in (('center_x', cx-2), ('center_y', cy-2),
                                   ('left_x', cx-2), ('left_y', cy-2),
                                   ('right_x', cx-2), ('right_y', cy-2)):
                    if sample[key] is not None:
                        sample[key] += int(delta)
            measurements.append(result)
        if not measurements:
            raise NoAdhesiveError('Detected adhesive is too small to measure reliably.')
        main = max(measurements, key=lambda item: item['length_px'])
        widths = [s['width_px'] for s in main['samples'] if s['valid']]
        # How long the line is that adhesive was looked for along: the panel edge,
        # averaged so that its pixel steps do not add length. Measured adhesive is
        # compared with it to tell a complete line from a partial one.
        edge = smoothed_path(np.asarray(line['original_xy'], dtype=float), 101)
        band_length = float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum())
        return dict(mask=mask, overlay=overlay, line_count=line_count, selected_line=index,
                    length_px=sum(m['length_px'] for m in measurements),
                    longest_length_px=main['length_px'], component_count=len(measurements),
                    mean_width_px=float(np.mean(widths)) if widths else None,
                    segments=main['segments'], components=measurements,
                    # seconds counts from the start of the capture, so for a later
                    # line it includes the lines before it; line_seconds is this line alone.
                    selected_boundary=line, seconds=perf_counter()-started,
                    line_seconds=perf_counter()-line_started,
                    band_length_px=band_length)
