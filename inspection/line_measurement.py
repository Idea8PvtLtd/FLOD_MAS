"""Calibrate the new refolded centreline and boundary intersections."""
import copy
import math
import numpy as np
from measurement_adjustment import current_ratio

# Segment widths are graded at the precision they are reported in.
GRADE_STEP_MM = 0.1
_EPS = 1e-9


def _to_step(value, direction):
    """value on the 0.1 mm grid: nearest (half up), 'down' or 'up'."""
    scaled = value / GRADE_STEP_MM
    whole = (math.floor(scaled + _EPS) if direction == 'down' else
             math.ceil(scaled - _EPS) if direction == 'up' else
             math.floor(scaled + 0.5 + _EPS))
    return round(whole * GRADE_STEP_MM, 6)


def grade_segments(segments, target, tolerance):
    """Grade one glue line's segments (in order along the line) on rounded widths.

    Each width is rounded to 0.1 mm and passes inside target ± tolerance. A
    segment that fails after rounding takes its rounding direction from the
    segments on either side: down when they are lower, up when higher. Several
    failing segments in a row are one run and share the direction set by the two
    segments beside the whole run. A side that is an end zone or has no width is
    skipped; at the end of the line the one side there is decides. Sets
    `graded_width_mm`, `rounding` and `within_tolerance` on every segment.
    """
    low, high = target - tolerance, target + tolerance
    inside = lambda value: low - _EPS <= value <= high + _EPS
    measured = [s for s in segments if s.get('average_width_mm') is not None]
    for segment in segments:
        segment['graded_width_mm'] = segment['rounding'] = segment['within_tolerance'] = None
    for segment in measured:
        segment['graded_width_mm'] = _to_step(segment['average_width_mm'], 'nearest')
        segment['rounding'] = 'nearest'
    failing = [not inside(s['graded_width_mm']) for s in measured]

    def side(index, step):
        while 0 <= index < len(measured) and measured[index].get('end_zone'):
            index += step
        return measured[index] if 0 <= index < len(measured) and not failing[index] else None

    start = 0
    while start < len(measured):
        if not failing[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(measured) and failing[end + 1]:
            end += 1
        sides = [s for s in (side(start - 1, -1), side(end + 1, 1)) if s is not None]
        if sides:
            reference = float(np.mean([s['average_width_mm'] for s in sides]))
            for segment in measured[start:end + 1]:
                value = segment['average_width_mm']
                if abs(reference - value) > _EPS:
                    direction = 'down' if reference < value else 'up'
                    segment['graded_width_mm'] = _to_step(value, direction)
                    segment['rounding'] = direction
        start = end + 1
    for segment in measured:
        segment['within_tolerance'] = inside(segment['graded_width_mm'])
    return segments


def _grade_component(component, target, tolerance):
    """Segment widths from the samples' widths, graded and reported at 0.1 mm.

    Only the final values are kept: each segment's width is the graded value
    (rounded, and up or down as its sides decide), and its minimum and maximum are
    rounded to the same 0.1 mm. The samples keep their widths, which the segment
    figures are computed from again when the distance ratio changes.
    """
    for segment in component['segments']:
        widths = [s['width_mm'] for s in component['samples']
                  if s['segment'] == segment['segment'] and s.get('width_mm') is not None]
        for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
            segment[f'{name}_width_mm'] = float(operation(widths)) if widths else None
    grade_segments(component['segments'], target, tolerance)
    for segment in component['segments']:
        segment['average_width_mm'] = segment.pop('graded_width_mm')
        for name in ('minimum_width_mm', 'maximum_width_mm'):
            if segment.get(name) is not None:
                segment[name] = _to_step(segment[name], 'nearest')


def _round_mm(value):
    return None if value is None else _to_step(value, 'nearest')


def rounded_record(measurement):
    """The measurement as saved: every millimetre figure at 0.1 mm.

    Segment widths already are final values. The running result keeps the
    samples' and the line's widths and lengths exact, for regrading when the
    distance ratio changes; the saved record holds only the rounded values.
    """
    record = copy.deepcopy(measurement)
    for name in ('length_mm', 'longest_length_mm', 'mean_width_mm'):
        record[name] = _round_mm(record.get(name))
    for component in record.get('components', ()):
        component['length_mm'] = _round_mm(component.get('length_mm'))
        for sample in component.get('samples', ()):
            sample['width_mm'] = _round_mm(sample.get('width_mm'))
    return record


def with_distance_ratio(measurement, ratio):
    """Recalculate a displayed result from its recorded ratio, without mutation."""
    from measurement_adjustment import validate_ratio
    ratio = validate_ratio(ratio)
    factor = ratio / validate_ratio(measurement.get('distance_ratio', 1.0))
    result = copy.deepcopy(measurement)
    def multiply(item, names):
        for name in names:
            if item.get(name) is not None:
                item[name] *= factor
    multiply(result, ('length_mm', 'longest_length_mm', 'mean_width_mm'))
    grades = []
    for component in result['components']:
        multiply(component, ('length_mm',))
        for sample in component['samples']:
            multiply(sample, ('width_mm',))
        # Segment figures are final rounded values, so they are worked out again
        # from the samples rather than scaled.
        _grade_component(component, result['target_width_mm'], result['tolerance_mm'])
        grades += [segment['within_tolerance'] for segment in component['segments']]
    main = max(result['components'], key=lambda c: c['length_px'])
    result['segments'] = main['segments']
    result['distance_ratio'] = ratio
    result['status'] = ('WARNING' if not result['metric'] or any(g is None for g in grades)
                        else 'PASS' if all(grades) else 'FAIL')
    return result


def measurement_record(result, scale, target_width, tolerance, end_exclusion_percent=0.0,
                       end_exclusion_px=None, end_trim_px=0.0):
    """Keep reference pixel measurements; map points, never use a global px scale."""
    if (isinstance(end_exclusion_percent, bool) or not np.isfinite(end_exclusion_percent)
            or not 0 <= end_exclusion_percent < 50):
        raise ValueError('End exclusion must be at least 0 and less than 50 percent')
    for value in (end_exclusion_px, end_trim_px):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError('End exclusion must be at least 0 pixels')
    ratio = current_ratio()
    components = copy.deepcopy(result['components'])
    for component in components:
        samples = component['samples']
        length = component['length_px']
        # In pixel mode the rounded tip cap is ignored up to end_trim_px from each
        # end; the end bar is drawn there and the end zone starts there.
        trim = float(end_trim_px or 0.0) if end_exclusion_px is not None else 0.0
        # Samples are about 5 px apart; cut half a step short of the trim so the
        # sample nearest the setting is kept and the end bar sits on the setting
        # rather than up to a step further in.
        edge = max(0.0, trim - length / max(1, len(samples) - 1) / 2) if trim else 0.0
        # A fixed pixel distance along the centreline, when set, replaces the percentage.
        if end_exclusion_px is not None:
            low = max(float(end_exclusion_px), edge)
        else:
            low = length * end_exclusion_percent / 100.0
        high = length - low
        # Shorter than both ends together: nothing is left to grade, so WARNING.
        empty = high <= low
        component['excluded_each_end_px'] = low
        component['trimmed_each_end_px'] = trim
        component['inspected_length_px'] = max(0.0, high-low)
        count = len(component['segments'])
        # In pixel mode the two end zones are measured and graded as their own
        # segments (0 before the equal split, count+1 after it) instead of ignored.
        graded_ends = end_exclusion_px is not None and low > trim and not empty
        for sample in samples:
            arc = sample['arc_px']
            trimmed = arc < edge or arc > length - edge
            excluded = empty or not low <= arc <= high
            sample['excluded_end'] = excluded and (trimmed or not graded_ends)
            if excluded and graded_ends and not trimmed:
                sample['segment'] = 0 if arc < low else count + 1
            elif excluded:
                # -1 in pixel mode, so an ignored tip can never join end zone 0.
                sample.update(segment=-1 if end_exclusion_px is not None else 0,
                              valid=False, width_px=None)
            elif low:
                sample['segment'] = min(count, int((arc-low)/(high-low)*count)+1)
        for segment in component['segments']:
            span = max(0.0, high-low)
            segment['start_arc_px'] = low+(segment['segment']-1)*span/count
            segment['end_arc_px'] = low+segment['segment']*span/count
        if graded_ends:
            component['segments'] = (
                [{'segment': 0, 'end_zone': 'start', 'start_arc_px': trim, 'end_arc_px': low}]
                + component['segments']
                + [{'segment': count + 1, 'end_zone': 'end', 'start_arc_px': high,
                    'end_arc_px': length - trim}])
        for segment in component['segments']:
            pixels = [s['width_px'] for s in samples if s['segment'] == segment['segment'] and s['valid']]
            segment['sample_count'] = len(pixels)
            for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
                segment[f'{name}_width_px'] = float(operation(pixels)) if pixels else None
        if scale is not None:
            centres = scale.to_mm([[s['center_x'], s['center_y']] for s in samples])
            component['length_mm'] = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()) * ratio
        else:
            component['length_mm'] = None
        for sample in samples:
            sample['width_mm'] = None
            if scale is not None and sample['valid']:
                endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                         [sample['right_x'], sample['right_y']]])
                width = float(np.linalg.norm(endpoints[1] - endpoints[0])) * ratio
                if np.isfinite(width):
                    sample['width_mm'] = width
        _grade_component(component, target_width, tolerance)
    main = max(components, key=lambda c: c['length_px'])
    widths = [s['width_mm'] for s in main['samples'] if s['width_mm'] is not None]
    pixel_widths = [s['width_px'] for s in main['samples'] if s['valid']]
    graded = [s['within_tolerance'] for component in components for s in component['segments']]
    metric = scale is not None and bool(widths)
    # A segment that could not be measured is reported as WARNING, naming no
    # verdict of its own. PASS/FAIL are reserved for fabrics whose every segment
    # was actually graded, so an ungradeable segment can never read as a failure.
    status = ('WARNING' if not metric or any(s is None for s in graded)
              else 'PASS' if all(graded) else 'FAIL')
    return dict(status=status, metric=metric, distance_ratio=ratio,
                line_count=result['line_count'], selected_line=result['selected_line'],
                component_count=result['component_count'], length_px=result['length_px'],
                longest_length_px=result['longest_length_px'],
                mean_width_px=float(np.mean(pixel_widths)) if pixel_widths else None,
                end_exclusion_percent=float(end_exclusion_percent),
                end_exclusion_px=None if end_exclusion_px is None else float(end_exclusion_px),
                end_trim_px=None if end_exclusion_px is None else float(end_trim_px or 0.0),
                length_mm=sum(c['length_mm'] for c in components) if scale is not None else None,
                longest_length_mm=main['length_mm'],
                mean_width_mm=float(np.mean(widths)) if widths else None,
                target_width_mm=float(target_width), tolerance_mm=float(tolerance),
                segments=main['segments'], components=components,
                selected_boundary=result['selected_boundary'], seconds=result['seconds'],
                coordinate_space='undistorted input frame',
                segment_basis='ten equal pixel arc-length intervals of the retained centreline in each component',
                length_basis='sum of separately measured adhesive components; no gap bridging')
