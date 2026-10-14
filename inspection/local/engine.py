"""In-memory source-line inspection with shared source detection per capture."""
from time import perf_counter
import numpy as np
from .geometry import (cv2, extract_lines, filter_wave_lines, offset_lines,
    region_mask, original_dimensions, unfold_maps, enhance_image, clean_mask,
    individual_crop, mapped_mask, smoothed_path)
from .width_measurement import measure_component


class NoAdhesiveError(ValueError):
    """The band of a line holds no adhesive that can be measured."""


class NoSourceLinesError(ValueError):
    """The source mask of a photo holds no line that can be inspected."""


class LiveInspection:
    segments = 10  # instances built without __init__ (the tests) keep the reference ten

    def __init__(self, source_path, glue_path, offset=100., source_width=1152, smooth_normal_rows=0,
                 smooth_logits=False, min_line_length_ratio=0., ribbon_finish=False, segments=10):
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
        # False measures the adhesive model's mask as it is. True replaces the mask of
        # a line found in one piece by a smooth ribbon fitted to the mask's edges and
        # to the band's own edges in the photo; see inspection/local/ribbon.
        self.ribbon_finish = bool(ribbon_finish)
        # How many equal segments the graded length of a line is split into.
        self.segments = int(segments)
        if self.segments < 1:
            raise ValueError('Segment count must be a whole number of at least 1')
        if self.ribbon_finish:
            from .ribbon import bead_ml
            bead_ml.load()      # a missing package or model file is reported at start, not at the first capture

    def run(self, photo, progress=lambda text: None):
        """Single-line callers want the rightmost line and nothing else."""
        return next(self.run_lines(photo, progress, only_rightmost=True))

    def find_lines(self, photo, progress=lambda text: None, sideways=False):
        """The source mask of a photo and the lines accepted in it, left to right.

        ``sideways`` also looks for pieces laid a quarter turn round; see
        ``_with_sideways``. Both the hand mode (``run_lines``) and the scene mode use it.
        """
        progress('AI source mask')
        source = self.source.predict(photo)
        lines = self._lines_in(source)
        if sideways:
            lines = self._with_sideways(source, lines)
        if not lines:
            raise NoSourceLinesError('No accepted source lines. Check placement or the source mask model.')
        return source, self._kept_in_order(lines)

    # A piece of the source mask this much wider than tall may lie sideways.
    SIDEWAYS_ASPECT = 1.5

    def _with_sideways(self, source, lines):
        """``lines`` plus the lines of pieces that lie a quarter turn round.

        The line geometry reads seams that run down the photo. A piece laid
        sideways shows in the source mask all the same (measured on four stills:
        turning the mask already predicted found the same lines as predicting again
        on the turned photo, which costs a second run of the model), so only the
        geometry is run again, on the wide pieces of the mask alone and turned a
        quarter. The lines found are turned back at once: everything after this,
        the band, the adhesive mask and the measuring, works in the photo as it is,
        for a line running any way, and so millimetres come from true frame positions.

        A wide piece is read the way that gives its longest line: the short end of
        a sideways panel can pass for a line of its own in the upright reading (it
        did on one still), and it is then dropped for the long edge.

        An upright board has no wide piece and pays only for a look at a reduced mask.
        """
        def wide(stats, scale=1):
            return [i for i in range(1, len(stats))
                    if stats[i, cv2.CC_STAT_WIDTH] >= self.SIDEWAYS_ASPECT * stats[i, cv2.CC_STAT_HEIGHT]
                    and stats[i, cv2.CC_STAT_WIDTH] * scale >= .1 * max(source.shape)]
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(source[::4, ::4] > 127), 8)
        if not wide(stats, 4):
            return lines
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(source > 127), 8)
        chosen = wide(stats)
        if not chosen:
            return lines
        height, width = source.shape

        def piece_of(line):
            """The piece of the mask a line runs along, and how long the line is."""
            points = np.asarray(line['original_xy'], dtype=float)
            at = np.rint(points).astype(int)
            found = labels[np.clip(at[:, 1], 0, height-1), np.clip(at[:, 0], 0, width-1)]
            found = found[found > 0]
            return (int(np.bincount(found).argmax()) if len(found) else 0), float(np.ptp(points, axis=0).max())
        turned = cv2.rotate(np.uint8(np.isin(labels, chosen)) * 255, cv2.ROTATE_90_CLOCKWISE)
        try:
            found = self._lines_in(turned)
        except ValueError:
            return lines        # the upright lines are still good
        for line in found:
            # A quarter turn clockwise puts photo point (x, y) at (height-1-y, x).
            for key in ('original_xy', 'offset_xy'):
                points = np.asarray(line[key], dtype=float)
                line[key] = np.column_stack((points[:, 1], height - 1 - points[:, 0])).tolist()
            line['turned'] = 'quarter turn'
        upright, sideways = [piece_of(line) for line in lines], [piece_of(line) for line in found]

        def longest(pieces, piece):
            return max((length for owner, length in pieces if owner == piece), default=0.)
        return ([line for line, (piece, _) in zip(lines, upright)
                 if piece not in chosen or longest(upright, piece) >= longest(sideways, piece)]
                + [line for line, (piece, _) in zip(found, sideways)
                   if longest(sideways, piece) > longest(upright, piece)])

    def _lines_in(self, source):
        """Every accepted line of a source mask, each with its inward band."""
        chains = extract_lines(source, hook_zone=.04, hook_angle=65)
        waves = np.zeros_like(source)
        for chain in chains:
            cv2.polylines(waves, [chain], False, 255, 1, cv2.LINE_8)
        filtered, _, _ = filter_wave_lines(waves, source, .008, .1, .9, make_preview=False)
        _, _, lines = offset_lines(filtered, self.offset, 'inside', source_mask=source, make_preview=False,
                                   smooth_rows=getattr(self, 'smooth_normal_rows', 0))
        return lines

    def _kept_in_order(self, lines):
        """Short false lines dropped (min_line_length_ratio), the rest left to right."""
        ratio = getattr(self, 'min_line_length_ratio', 0.)
        if ratio > 0 and len(lines) > 1:
            lengths = [float(np.ptp(np.asarray(line['original_xy'], dtype=float), axis=0).max()) for line in lines]
            lines = [line for line, length in zip(lines, lengths) if length >= ratio * max(lengths)]
        lines.sort(key=lambda line: float(np.mean(np.asarray(line['original_xy'])[:, 0])))
        return lines

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
        source, lines = self.find_lines(photo, progress, sideways=True)
        if lines and all(line.get('turned') for line in lines):
            # Fabric laid sideways: its lines are stacked down the photo, so they are
            # numbered top first, the order they are placed in (as on main, where the
            # whole capture was turned a quarter for detection).
            lines.sort(key=lambda line: float(np.mean(np.asarray(line['original_xy'])[:, 1])))
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
        yield from self.measure_found(photo, source, [(lines[index], index+1) for index in indices],
                                      total, progress, on_error, started=started)

    def measure_found(self, photo, source, picks, total, progress=lambda text: None, on_error=None,
                      started=None):
        """Measure chosen lines of a photo whose lines were found by ``find_lines``.

        ``picks`` = [(line, number), ...]; yields the same record as ``run_lines``,
        with ``selected_line`` = the number given and ``line_count`` = ``total``.
        A line that fails goes to ``on_error(number, error)`` and the rest are
        still measured; without ``on_error`` the error is raised.
        """
        started = perf_counter() if started is None else started
        try:
            for line, number in picks:
                progress(f'Inspecting line {number} of {total}')
                try:
                    yield self._inspect_line(photo, source, line, number, total, started, progress)
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(number, error)
        finally:
            # The enhanced photo is kept only while this capture's lines are measured.
            self._enhanced = None
            self._region = None

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
        measurements = self._measure(mask)
        if not measurements:
            raise NoAdhesiveError('Detected adhesive is too small to measure reliably.')
        ribbon_record = None
        if getattr(self, 'ribbon_finish', False):
            progress('Fitting the smooth ribbon')
            mask, measurements, ribbon_record = self._ribbon(photo, source, mask, measurements)
        overlay = photo.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        for result in measurements:
            points = np.array([[s['center_x'], s['center_y']] for s in result['samples']])
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], False, (0, 255, 255), 2)
        main = max(measurements, key=lambda item: item['length_px'])
        widths = [s['width_px'] for s in main['samples'] if s['valid']]
        # How long the line is that adhesive was looked for along: the panel edge,
        # averaged so that its pixel steps do not add length. Measured adhesive is
        # compared with it to tell a complete line from a partial one.
        edge = smoothed_path(np.asarray(line['original_xy'], dtype=float), 101)
        band_length = float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum())
        record = dict(mask=mask, overlay=overlay, line_count=line_count, selected_line=index,
                    length_px=sum(m['length_px'] for m in measurements),
                    longest_length_px=main['length_px'], component_count=len(measurements),
                    mean_width_px=float(np.mean(widths)) if widths else None,
                    segments=main['segments'], components=measurements,
                    # seconds counts from the start of the capture, so for a later
                    # line it includes the lines before it; line_seconds is this line alone.
                    selected_boundary=line, seconds=perf_counter()-started,
                    line_seconds=perf_counter()-line_started,
                    band_length_px=band_length)
        if ribbon_record is not None:
            record['ribbon'] = ribbon_record
        return record

    def _ribbon(self, photo, source, mask, measurements):
        """The mask and its measurements after the ribbon, and what the ribbon did.

        A line in several pieces is left as it is: a ribbon is one continuous
        strip and would cover the gap that the coverage and gap checks look for.
        Whenever no ribbon is used the mask's own measurements are returned.
        """
        from . import ribbon
        if len(measurements) != 1:
            return mask, measurements, dict(state='skipped', reason='line is in several pieces')
        samples = [s for s in measurements[0]['samples'] if s['valid']]
        raw_width = float(np.mean([s['width_px'] for s in samples])) if samples else None
        region = getattr(self, '_region', None)
        if region is None or region[0] is not photo:
            region = self._region = (photo, ribbon.glue_region(photo, source))
        try:
            found = ribbon.finish(photo, region[1], mask, [[s['center_x'], s['center_y']] for s in samples])
        except Exception as error:      # the mask's own measurement stands
            return mask, measurements, dict(state='refused', reason=f'ribbon failed: {error}', mask_width_px=raw_width)
        ring = found.pop('ring', None)
        found['mask_width_px'] = raw_width
        if ring is None:
            return mask, measurements, found
        filled = ribbon.ring_mask(mask.shape, ring)
        # Only the part of the photo the ribbon lies in is searched for it.
        x, y, w, h = cv2.boundingRect(np.round(ring).astype(np.int32))
        x, y = max(0, x-2), max(0, y-2)
        finished = self._measure(filled[y:y+h+4, x:x+w+4], origin=(x, y))
        if len(finished) != 1:
            return mask, measurements, dict(found, state='refused', reason='ribbon could not be measured')
        return filled, finished, found

    def _measure(self, mask, origin=(0, 0)):
        """Each real component of a mask, measured. Disconnected glue is not bridged across gaps.

        ``origin`` is where ``mask`` lies in the photo when it is a part cut from it."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        measurements = []
        for i in range(1, count):
            cx, cy, cw, ch, area = stats[i]
            if area < 100:
                continue
            component = np.uint8(labels[cy:cy+ch, cx:cx+cw] == i)*255
            component = np.pad(component, 2)
            try:
                result = measure_component(component, segments=self.segments)
            except ValueError:
                continue
            for sample in result['samples']:
                dx, dy = cx-2+origin[0], cy-2+origin[1]
                for key, delta in (('center_x', dx), ('center_y', dy),
                                   ('left_x', dx), ('left_y', dy),
                                   ('right_x', dx), ('right_y', dy)):
                    if sample[key] is not None:
                        sample[key] += int(delta)
            measurements.append(result)
        return measurements
