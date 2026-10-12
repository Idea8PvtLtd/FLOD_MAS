"""Calibrate the new refolded centreline and boundary intersections."""
import copy
import numpy as np
from measurement_adjustment import current_ratio


# Extra checks, all off at 0 so that grading is unchanged unless they are set
# (config: inspection.<name>).
#   local_defect_length_mm        a run of consecutive samples outside the tolerance
#                                 that is at least this long fails its segments, even
#                                 when the segment averages are inside the tolerance
#   min_coverage_percent          the measured length must be at least this share of
#                                 the length of the band the glue was looked for in
#   max_gap_mm                    the largest gap between two pieces of one line
#   ignore_pieces_shorter_than_mm stray pieces shorter than this are not graded (the
#                                 longest piece always is)
CHECKS = {'local_defect_length_mm': 0.0, 'min_coverage_percent': 0.0,
          'max_gap_mm': 0.0, 'ignore_pieces_shorter_than_mm': 0.0}


def grading_checks(settings):
    """The extra checks from the ``inspection`` section of the configuration."""
    checks = dict(CHECKS)
    for name in CHECKS:
        value = (settings or {}).get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'inspection.{name} must be zero or a positive number')
        checks[name] = float(value)
    if checks['min_coverage_percent'] > 100:
        raise ValueError('inspection.min_coverage_percent must be at most 100')
    return checks


def _local_defects(component, low, high, minimum_mm):
    """Runs of consecutive valid samples outside [low, high] at least ``minimum_mm`` long."""
    samples = component['samples']
    arcs = [s.get('arc_mm') for s in samples]
    known = [b - a for a, b in zip(arcs, arcs[1:]) if a is not None and b is not None]
    step = float(np.median(known)) if known else 0.0
    found, run, kind = [], [], None

    def close():
        if not run:
            return
        first, last = samples[run[0]], samples[run[-1]]
        if first.get('arc_mm') is None or last.get('arc_mm') is None:
            return
        # A run of n samples covers n spacings: half a spacing beyond each end sample.
        length = last['arc_mm'] - first['arc_mm'] + step
        if length + 1e-9 >= minimum_mm:
            widths = [samples[i]['width_mm'] for i in run]
            found.append({'kind': kind, 'length_mm': float(length),
                          'start_arc_px': float(first['arc_px']), 'end_arc_px': float(last['arc_px']),
                          'extreme_width_mm': float(max(widths) if kind == 'wide' else min(widths)),
                          'segments': sorted({samples[i]['segment'] for i in run})})

    for index, sample in enumerate(samples):
        width = sample.get('width_mm')
        current = None if width is None else 'wide' if width > high else 'narrow' if width < low else None
        if current != kind:
            close()
            run, kind = [], current
        if current is not None:
            run.append(index)
    close()
    return found


def _grade(result):
    """Grade a measurement in place from its millimetre values and stored checks."""
    checks = result.get('checks') or CHECKS
    low = result['target_width_mm'] - result['tolerance_mm']
    high = result['target_width_mm'] + result['tolerance_mm']
    components = result['components']
    main = max(components, key=lambda c: c['length_px'])
    shortest = checks['ignore_pieces_shorter_than_mm']
    grades, defects, reasons = [], [], []
    for component in components:
        length = component.get('length_mm')
        component['ignored'] = bool(shortest > 0 and component is not main
                                    and length is not None and length < shortest)
        found = ([] if component['ignored'] or checks['local_defect_length_mm'] <= 0 else
                 _local_defects(component, low, high, checks['local_defect_length_mm']))
        component['local_defects'] = found
        marked = {segment for defect in found for segment in defect['segments']}
        for segment in component['segments']:
            average = segment.get('average_width_mm')
            segment['within_tolerance'] = None if average is None else low <= average <= high
            segment['local_defect'] = segment['segment'] in marked
            if segment['local_defect']:
                segment['within_tolerance'] = False
            if not component['ignored']:
                grades.append(segment['within_tolerance'])
        defects += found
    if defects:
        worst = max(defects, key=lambda d: d['length_mm'])
        reasons.append(f"{len(defects)} place(s) outside tolerance for {checks['local_defect_length_mm']:g} mm or more "
                       f"(longest {worst['length_mm']:.1f} mm, {worst['kind']}, {worst['extreme_width_mm']:.2f} mm)")
    structural = False
    coverage = result.get('coverage_percent')
    if checks['min_coverage_percent'] > 0 and coverage is not None and coverage < checks['min_coverage_percent']:
        structural = True
        reasons.append(f"adhesive found on {coverage:.0f} % of the line, below {checks['min_coverage_percent']:g} %")
    gap = result.get('largest_gap_mm')
    if checks['max_gap_mm'] > 0 and gap is not None and gap > checks['max_gap_mm']:
        structural = True
        reasons.append(f"gap of {gap:.1f} mm in the line, above {checks['max_gap_mm']:g} mm")
    result['segments'] = main['segments']
    result['local_defects'] = defects
    result['fail_reasons'] = reasons
    # A segment that could not be measured is reported as WARNING, naming no
    # verdict of its own. PASS/FAIL are reserved for fabrics whose every segment
    # was actually graded, so an ungradeable segment can never read as a failure.
    # A line that is incomplete (coverage, gap) is a failure whatever its segments say.
    result['status'] = ('FAIL' if structural
                        else 'WARNING' if not result['metric'] or any(g is None for g in grades)
                        else 'PASS' if all(grades) else 'FAIL')
    return result


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
    multiply(result, ('length_mm', 'longest_length_mm', 'mean_width_mm', 'largest_gap_mm'))
    for component in result['components']:
        multiply(component, ('length_mm',))
        for sample in component['samples']:
            multiply(sample, ('width_mm', 'arc_mm'))
        for segment in component['segments']:
            multiply(segment, ('average_width_mm', 'minimum_width_mm', 'maximum_width_mm'))
    result['distance_ratio'] = ratio
    return _grade(result)


def _gaps(components, scale, ratio):
    """Pieces ordered along the line and the gaps between neighbouring ends (px, mm)."""
    ends = []
    for component in components:
        centres = [(s['center_x'], s['center_y']) for s in component['samples']]
        if len(centres) >= 2:
            ends.append((np.asarray(centres[0], float), np.asarray(centres[-1], float)))
    if len(ends) < 2:
        return 0.0, (0.0 if scale is not None else None)
    # Every piece runs the same way along the line (the measurement orients its
    # centreline by the image axis), so the first ends give the order.
    axis = int(np.argmax(np.ptp(np.array([e[0] for e in ends] + [e[1] for e in ends]), axis=0)))
    ends.sort(key=lambda pair: min(pair[0][axis], pair[1][axis]))
    largest_px, largest_mm = 0.0, (0.0 if scale is not None else None)
    for (_, before), (after, _) in zip(ends, ends[1:]):
        candidates = [(before, after)]
        largest_px = max(largest_px, float(np.linalg.norm(after - before)))
        if scale is not None:
            a, b = scale.to_mm([before, after])
            largest_mm = max(largest_mm, float(np.linalg.norm(b - a)) * ratio)
    return largest_px, largest_mm


def measurement_record(result, scale, target_width, tolerance, end_exclusion_percent=0.0, checks=None):
    """Keep reference pixel measurements; map points, never use a global px scale.

    ``checks`` are the extra grading checks (see CHECKS); None or all zero grades
    exactly as before, by segment averages alone.
    """
    checks = dict(CHECKS, **(checks or {}))
    if (isinstance(end_exclusion_percent, bool) or not np.isfinite(end_exclusion_percent)
            or not 0 <= end_exclusion_percent < 50):
        raise ValueError('End exclusion must be at least 0 and less than 50 percent')
    ratio = current_ratio()
    components = copy.deepcopy(result['components'])
    for component in components:
        samples = component['samples']
        low = component['length_px'] * end_exclusion_percent / 100.0
        high = component['length_px'] - low
        component['excluded_each_end_px'] = low
        component['inspected_length_px'] = high-low
        count = len(component['segments'])
        for sample in samples:
            excluded = not low <= sample['arc_px'] <= high
            sample['excluded_end'] = excluded
            if excluded:
                sample.update(segment=0, valid=False, width_px=None)
            elif end_exclusion_percent:
                sample['segment'] = min(count, int((sample['arc_px']-low)/(high-low)*count)+1)
        for segment in component['segments']:
            segment['start_arc_px'] = low+(segment['segment']-1)*(high-low)/count
            segment['end_arc_px'] = low+segment['segment']*(high-low)/count
            pixels = [s['width_px'] for s in samples if s['segment'] == segment['segment'] and s['valid']]
            segment['sample_count'] = len(pixels)
            for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
                segment[f'{name}_width_px'] = float(operation(pixels)) if pixels else None
        if scale is not None:
            centres = scale.to_mm([[s['center_x'], s['center_y']] for s in samples])
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
            component['length_mm'] = float(steps.sum()) * ratio
            along = np.concatenate(([0.], np.cumsum(steps))) * ratio
            for sample, position in zip(samples, along):
                sample['arc_mm'] = float(position) if np.isfinite(position) else None
        else:
            component['length_mm'] = None
            for sample in samples:
                sample['arc_mm'] = None
        for sample in samples:
            sample['width_mm'] = None
            if scale is not None and sample['valid']:
                endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                         [sample['right_x'], sample['right_y']]])
                width = float(np.linalg.norm(endpoints[1] - endpoints[0])) * ratio
                if np.isfinite(width):
                    sample['width_mm'] = width
        for segment in component['segments']:
            widths = [s['width_mm'] for s in samples
                      if s['segment'] == segment['segment'] and s['width_mm'] is not None]
            for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
                segment[f'{name}_width_mm'] = float(operation(widths)) if widths else None
    main = max(components, key=lambda c: c['length_px'])
    widths = [s['width_mm'] for s in main['samples'] if s['width_mm'] is not None]
    pixel_widths = [s['width_px'] for s in main['samples'] if s['valid']]
    metric = scale is not None and bool(widths)
    band = result.get('band_length_px')
    covered = sum(c['length_px'] for c in components)
    coverage = (float(min(100.0, 100.0 * covered / band))
                if band is not None and np.isfinite(band) and band > 0 else None)
    gap_px, gap_mm = _gaps(components, scale, ratio)
    record = dict(status=None, metric=metric, distance_ratio=ratio,
                line_count=result['line_count'], selected_line=result['selected_line'],
                component_count=result['component_count'], length_px=result['length_px'],
                longest_length_px=result['longest_length_px'],
                mean_width_px=float(np.mean(pixel_widths)) if pixel_widths else None,
                end_exclusion_percent=float(end_exclusion_percent),
                length_mm=sum(c['length_mm'] for c in components) if scale is not None else None,
                longest_length_mm=main['length_mm'],
                mean_width_mm=float(np.mean(widths)) if widths else None,
                target_width_mm=float(target_width), tolerance_mm=float(tolerance),
                segments=main['segments'], components=components,
                selected_boundary=result['selected_boundary'], seconds=result['seconds'],
                coordinate_space='undistorted input frame',
                segment_basis='ten equal pixel arc-length intervals of the retained centreline in each component',
                length_basis='sum of separately measured adhesive components; no gap bridging',
                checks=checks, band_length_px=band, coverage_percent=coverage,
                largest_gap_px=gap_px, largest_gap_mm=gap_mm)
    return _grade(record)
