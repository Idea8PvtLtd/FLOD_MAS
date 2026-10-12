"""Width report: where each glue line is wider or narrower than a good line's natural
variation allows, in millimetres, as a page to look at.

    python tools/width_report.py --runs RUNS_DIR --out OUT_DIR

``RUNS_DIR`` holds one pickle per frame from an outlining run (``image`` path, packed
``roi``, ``shape``, ``colour``, ``frame`` and ``lines`` with their ``ring``). For every
line the width is read every 2 px, turned into millimetres with a scale fitted over
the frame, and the line is scanned with the empirical limits of
``segmentation.width_stats``.

The width is read from the segmentation mask when ``--mask-model`` names a trained
SegFormer (``segmentation.segformer_local``): pixel by pixel across the line, the
two ends placed between pixels. Without it, the band edges placed by the trained
trees are used. ``--held-out`` lists the runs the mask model was not trained on:
only those lines set the limits, since a model reads the frames it was trained on
more evenly than new ones.

The lines given are taken to be good ones of nominal width: they set the scale and
the limits. Limits for a line come from other frames (five groups by capture time),
so a line never sets its own limit.
"""
from __future__ import annotations

import argparse
import html
import importlib
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

import pipeline_steps as steps  # noqa: E402
from segmentation import bead_edges, edge_methods, segformer_local, width_stats  # noqa: E402

STEP = bead_edges.STEP                 # pixels between samples along a line
TIP = 30                               # samples left out at each end (the bead tapers there)
SIZES = tuple(int(size / STEP) for size in width_stats.WINDOWS_PX)
BLOCK = SIZES[0]
FOLDS = 5
MIN_LINES = 20                         # a colour with fewer good lines shares the limits of all colours
SHOW_MM = 0.5                          # colour scale of the overlay: +- this many mm
VERDICT = {'glitch': 'segmentation glitch, not the glue', 'confirmed': 'also seen in the band edges',
           'mask only': 'in the mask only, not confirmed'}
WIDER, NARROWER, GLITCH = (40, 40, 230), (230, 110, 30), (0, 215, 255)       # BGR


def read_lines(run, mask_model=None):
    """Per line of one frame: sample points, normals, the two edges and the width."""
    importlib.import_module('segmentation.bead_ml')
    image = cv2.imread(run['image'])
    roi = np.unpackbits(run['roi'])[:run['shape'][0] * run['shape'][1]].reshape(run['shape'])
    probability = mask_model.probability(image, roi) if mask_model is not None else None
    lines = []
    for number, line in enumerate(run['lines'], 1):
        if 'backed' not in line:
            continue
        found = bead_edges.measure(image, steps._centre(np.asarray(line['ring'])), roi, search=22,
                                   edge_method=edge_methods.METHODS['trees'])
        if found is None or found['span'].stop - found['span'].start < 2 * TIP + 4 * SIZES[-1]:
            continue
        keep = slice(found['span'].start + TIP, found['span'].stop - TIP)
        low, high = found['low'][keep], found['high'][keep]
        band_px = median_filter(high - low, 5, mode='nearest')        # read from the picture, not the mask
        missing = np.zeros(len(low), bool)
        sharp = np.ones(len(low))
        if probability is not None:
            # From the segmentation mask instead; a reading the mask does not give
            # is bridged from its neighbours, and a line mostly without mask is left out.
            low, high, sharp = segformer_local.mask_edges(probability, found['points'][keep],
                                                          found['normals'][keep], with_sharpness=True)
            missing = ~(np.isfinite(low) & np.isfinite(high))
            if missing.mean() > 0.1:
                continue
            samples = np.arange(len(low))
            low = np.interp(samples, samples[~missing], low[~missing])
            high = np.interp(samples, samples[~missing], high[~missing])
            sharp = np.interp(samples, samples[~missing], sharp[~missing])
        lines.append({'frame': run['frame'], 'colour': run['colour'], 'number': number,
                      'points': found['points'][keep], 'normals': found['normals'][keep],
                      'low': median_filter(low, 5, mode='nearest'), 'high': median_filter(high, 5, mode='nearest'),
                      'state': found['state'][keep], 'band_px': band_px, 'missing': missing,
                      'sharp': median_filter(sharp, 5, mode='nearest')})
        lines[-1]['width_px'] = lines[-1]['high'] - lines[-1]['low']
        lines[-1]['middle_px'] = (lines[-1]['high'] + lines[-1]['low']) / 2.0
    return image, lines


def colour_of(mm):
    """Blue for narrower, red for wider, pale near the line's own width."""
    amount = float(np.clip(mm / SHOW_MM, -1.0, 1.0))
    base = np.array(WIDER if amount > 0 else NARROWER, np.float64)
    return tuple(int(value) for value in (255 + (base - 255) * abs(amount)))


def draw(image, lines):
    """The frame with each line's band coloured by its departure and findings outlined."""
    shade = image.copy()
    for line in lines:
        left = line['points'] + line['normals'] * line['low'][:, None]
        right = line['points'] + line['normals'] * line['high'][:, None]
        for first in range(0, len(left) - 5, 5):
            last = first + 6
            quad = np.vstack([left[first:last], right[first:last][::-1]])
            cv2.fillPoly(shade, [np.round(quad).astype(np.int32)],
                         colour_of(float(np.mean(line['departure'][first:last]))))
    result = cv2.addWeighted(shade, 0.6, image, 0.4, 0)
    for line in lines:
        for finding in line['findings']:
            part = slice(finding['first'], finding['last'] + 1)
            colour = WIDER if finding['change'] > 0 else NARROWER
            thick = 5 if finding['level'] == 'significant' else 2
            for edge in ('low', 'high'):
                side = line['points'][part] + line['normals'][part] * (line[edge][part] + (-9 if edge == 'low' else 9))[:, None]
                cv2.polylines(result, [np.round(side).astype(np.int32)], False, colour, thick, cv2.LINE_AA)
            where = line['points'][(finding['first'] + finding['last']) // 2]
            cv2.putText(result, f"{finding['change']:+.2f}", (int(where[0]) + 40, int(where[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 4, cv2.LINE_AA)
        for first, last, _ in line['glitches']:
            part = slice(first, last + 1)
            for edge, away in (('low', -16), ('high', 16)):
                side = line['points'][part] + line['normals'][part] * (line[edge][part] + away)[:, None]
                cv2.polylines(result, [np.round(side).astype(np.int32)], False, GLITCH, 4, cv2.LINE_AA)
        top = line['points'][0]
        cv2.putText(result, f"{line['number']}: {line['median_mm']:.2f} mm", (int(top[0]) - 120, max(50, int(top[1]) - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 4, cv2.LINE_AA)
    return result


def chart(line, limits_mm, per_mm):
    """The line's width along its length as an SVG."""
    width, height, left, top = 760, 190, 46, 10
    along = np.arange(len(line['mm'])) * STEP / per_mm                  # mm along the line
    low, high = line['median_mm'] - 1.0, line['median_mm'] + 1.0

    def x(value):
        return left + (width - left - 8) * value / max(along[-1], 1e-6)

    def y(value):
        return top + (height - top - 26) * (high - np.clip(value, low, high)) / (high - low)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
             f'<rect x="{left}" y="{top}" width="{width - left - 8}" height="{height - top - 26}" fill="#fff" stroke="#bbb"/>']
    single = limits_mm['significant'][SIZES[0]]
    course = width_stats.block_means(line['trend'], 10)
    where = width_stats.block_means(along, 10)
    band = ' '.join(f'{x(a):.1f},{y(b + single):.1f}' for a, b in zip(where, course)) + ' ' + \
        ' '.join(f'{x(a):.1f},{y(b - single):.1f}' for a, b in zip(where[::-1], course[::-1]))
    parts.append(f'<polygon points="{band}" fill="#e3f1e3"/>')
    for finding in line['findings']:
        fill = '#f6c9c9' if finding['change'] > 0 else '#c9dbf6'
        opacity = 0.95 if finding['level'] == 'significant' else 0.5
        parts.append(f'<rect x="{x(along[finding["first"]]):.1f}" y="{top}" '
                     f'width="{max(2.0, x(along[finding["last"]]) - x(along[finding["first"]])):.1f}" '
                     f'height="{height - top - 26}" fill="{fill}" opacity="{opacity}"/>')
    for first, last, _ in line['glitches']:
        parts.append(f'<rect x="{x(along[first]):.1f}" y="{height - 34}" '
                     f'width="{max(3.0, x(along[last]) - x(along[first])):.1f}" height="8" fill="#e6b800"/>')
    if low < width_stats.NOMINAL_MM < high:
        parts.append(f'<line x1="{left}" x2="{width - 8}" y1="{y(width_stats.NOMINAL_MM):.1f}" '
                     f'y2="{y(width_stats.NOMINAL_MM):.1f}" stroke="#888"/>')
    parts.append('<polyline points="' + ' '.join(f'{x(a):.1f},{y(b):.1f}' for a, b in zip(where, course))
                 + '" fill="none" stroke="#2a7" stroke-dasharray="4 3"/>')
    blocks = width_stats.block_means(line['mm'], 5)
    places = width_stats.block_means(along, 5)
    points = ' '.join(f'{x(a):.1f},{y(b):.1f}' for a, b in zip(places, blocks))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#222" stroke-width="1.2"/>')
    for value in (low, line['median_mm'], high):
        parts.append(f'<text x="{left - 4}" y="{y(value) + 4:.1f}" font-size="11" text-anchor="end">{value:.1f}</text>')
    for value in np.arange(0, along[-1], 50):
        parts.append(f'<text x="{x(value):.1f}" y="{height - 10}" font-size="11" text-anchor="middle">{value:.0f}</text>')
    parts.append(f'<text x="{width - 8}" y="{height - 10}" font-size="11" text-anchor="end">mm along the line</text></svg>')
    return ''.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mask-model', help='trained SegFormer directory: read the width from its mask')
    parser.add_argument('--held-out', help='file listing the runs the mask model was not trained on')
    args = parser.parse_args(argv)
    mask_model = segformer_local.GlueMask(args.mask_model) if args.mask_model else None
    held = None
    if args.held_out:
        held = {Path(line.strip()).stem for line in Path(args.held_out).read_text().splitlines() if line.strip()}
    out = Path(args.out)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    runs = sorted(Path(args.runs).glob('*.pkl'))
    frames = []
    for path in runs:
        run = pickle.loads(path.read_bytes())
        image, lines = read_lines(run, mask_model)
        fresh = held is None or path.stem in held
        for line in lines:
            line['fresh'] = fresh
        frames.append({'frame': run['frame'], 'image': run['image'], 'colour': run['colour'],
                       'shape': image.shape, 'lines': lines, 'fresh': fresh})
        print(run['frame'], len(lines), flush=True)
    every = [line for frame in frames for line in frame['lines']]
    frame_size = (frames[0]['shape'][1], frames[0]['shape'][0])

    # 1. The scale: pixels per millimetre over the frame, per fabric colour.
    places = np.vstack([np.column_stack([width_stats.block_means(line['points'][:, 0], BLOCK),
                                         width_stats.block_means(line['points'][:, 1], BLOCK)]) for line in every])
    widths = np.concatenate([width_stats.block_means(line['width_px'], BLOCK) for line in every])
    groups = np.concatenate([[line['colour']] * (len(line['width_px']) // BLOCK) for line in every])
    scale = width_stats.fit_scale(places, widths, groups, frame_size)
    for line in every:
        line['per_mm'] = width_stats.pixels_per_mm(scale, line['points'], line['colour'])
        line['mm'] = line['width_px'] / line['per_mm']
        line['median_mm'] = float(np.median(line['mm']))
        line['trend'] = width_stats.slow_trend(line['mm'])
        line['departure'] = line['mm'] - line['trend']
        band_mm = line['band_px'] / line['per_mm']
        line['band_departure'] = band_mm - width_stats.slow_trend(band_mm)
        sideways = line['middle_px'] / line['per_mm']
        line['sideways'] = sideways - median_filter(sideways, 151, mode='nearest')

    # 2. Each colour's natural spread, so that the colours can share limits.
    spread = {}
    for colour in sorted({line['colour'] for line in every}):
        blocks = np.concatenate([width_stats.block_means(line['departure'], BLOCK)
                                 for line in every if line['colour'] == colour and line['fresh']])
        spread[colour] = 1.4826 * float(np.median(np.abs(blocks)))

    # 3. Limits from the other frames: five groups by capture time.
    order = sorted({frame['frame'] for frame in frames})
    fold_of = {name: min(FOLDS - 1, index * FOLDS // len(order)) for index, name in enumerate(order)}
    limits = {}

    def limits_for(colour, fold):
        # In units of the colour's spread. A colour with few lines borrows the
        # limits of all colours together.
        def chosen(only):
            return [line['departure'] / spread[line['colour']] for line in every
                    if line['fresh'] and fold_of[line['frame']] != fold and (not only or line['colour'] == colour)]
        own = chosen(True)
        return width_stats.limits_from_good_lines(own if len(own) >= MIN_LINES else chosen(False), SIZES)

    for fold in list(range(FOLDS)) + [None]:          # None: from every fresh line
        limits[fold] = {colour: limits_for(colour, fold) for colour in spread}
    offsets = {fold: np.percentile([abs(line['median_mm'] - width_stats.NOMINAL_MM) for line in every
                                    if line['fresh'] and fold_of[line['frame']] != fold], [80.0, 95.0])
               for fold in range(FOLDS)}
    fresh_lines = [line for line in every if line['fresh']]
    soft_limit = float(np.percentile(np.concatenate([line['sharp'] for line in fresh_lines]), 2.0))
    sideways_limits = {fold: width_stats.limits_from_good_lines(
        [line['sideways'] for line in fresh_lines if fold is None or fold_of[line['frame']] != fold], SIZES[:2])
        for fold in list(range(FOLDS)) + [None]}
    for line in every:
        fold = fold_of[line['frame']]
        line['findings'] = width_stats.scan(line['departure'] / spread[line['colour']],
                                            limits[fold][line['colour']], SIZES)
        for finding in line['findings']:
            finding['change'] *= spread[line['colour']]            # back to millimetres
        # Where the mask itself is in question: no mask, an end that fades out, or
        # the mask's middle stepping sideways. Limits again from the good lines.
        soft = line['sharp'] < soft_limit
        shifted = np.zeros(len(soft), bool)
        for step in width_stats.scan(line['sideways'], sideways_limits[fold], SIZES[:2]):
            if step['level'] == 'significant':
                shifted[step['first']:step['last'] + 1] = True
        line['glitches'] = ([(first, last, 'no mask') for first, last in width_stats.stretches(line['missing'], 3)]
                            + [(first, last, 'mask fades out') for first, last in width_stats.stretches(soft, 10)]
                            + [(first, last, 'mask steps sideways') for first, last in width_stats.stretches(shifted, 5)])
        doubtful = np.zeros(len(soft), bool)
        for first, last, _ in line['glitches']:
            doubtful[first:last + 1] = True
        line['doubtful'] = doubtful
        for finding in line['findings']:
            finding['verdict'] = width_stats.judge_change(finding, doubtful, line['band_departure'])
        off = abs(line['median_mm'] - width_stats.NOMINAL_MM)
        line['whole'] = 'significant' if off > offsets[fold][1] else ('notable' if off > offsets[fold][0] else '')

    # 4. The page.
    all_limits = limits[None]
    mean_per_mm = float(np.mean([np.mean(line['per_mm']) for line in every]))
    colours = sorted(spread)
    counts = {level: sum(any(f['level'] == level for f in line['findings']) for line in every)
              for level in ('notable', 'significant')}
    tally = {}
    for line in every:
        for finding in line['findings']:
            key = (finding['level'], finding['verdict'])
            tally[key] = tally.get(key, 0) + 1
    glitch_lines = sum(bool(line['glitches']) for line in every)
    verdict_text = (f'{glitch_lines} of {len(every)} lines have at least one stretch where the mask is in question '
                    f'({sum(len(line["glitches"]) for line in every)} stretches). Of the width changes marked significant: '
                    + ', '.join(f'{tally.get(("significant", key), 0)} {VERDICT[key]}' for key in VERDICT)
                    + '; of those marked notable: '
                    + ', '.join(f'{tally.get(("notable", key), 0)} {VERDICT[key]}' for key in VERDICT)
                    + '. Yellow marks in the pictures and under the charts show the glitch stretches.')
    rows = []
    for colour in colours:
        chosen = [line for line in every if line['colour'] == colour]
        rows.append(f"<tr><td>{colour}</td><td>{len(chosen)}</td>"
                    f"<td>{np.mean([np.mean(line['per_mm']) for line in chosen]):.2f}</td>"
                    f"<td>{np.mean([line['median_mm'] for line in chosen]):.2f} &plusmn; {np.std([line['median_mm'] for line in chosen]):.2f}</td>"
                    f"<td>{spread[colour]:.3f}</td>"
                    + ''.join(f"<td>&plusmn;{all_limits[colour]['significant'][size] * spread[colour]:.2f}</td>" for size in SIZES)
                    + f"<td>{sum(any(f['level'] == 'significant' for f in line['findings']) for line in chosen)} of {len(chosen)}</td></tr>")
    style = ('body{font:14px sans-serif;margin:16px;background:#f4f4f4;color:#111;max-width:1250px}'
             'table{border-collapse:collapse;background:#fff;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}'
             'th{background:#eee}img{max-width:100%;border:1px solid #999}.f{background:#fff;padding:10px;margin:14px 0;border:1px solid #ccc}'
             'h2{margin:4px 0}.w{color:#c22}.n{color:#26c}.s{font-weight:bold}.g{color:#8a6d00}')
    lengths = ''.join(f'<th>{size * STEP:.0f} px &asymp; {size * STEP / mean_per_mm:.0f} mm</th>' for size in SIZES)
    page = [f'<!doctype html><meta charset="utf-8"><title>Glue line width report</title><style>{style}</style>',
            '<h1>Glue line width: where it increases and decreases</h1>',
            f'<p>{len(frames)} frames, {len(every)} glue lines. The strip is taken to be '
            f'{width_stats.NOMINAL_MM:g} mm wide on average; the camera is not calibrated, so the scale is fitted from '
            f'that: <b>{mean_per_mm:.2f} pixels per mm on average</b>, varying over the frame and by fabric colour '
            f'(the fitted surface explains {scale["explained"]:.0%} of the variation in width between places).</p>',
            '<h2>Method</h2><ol>'
            '<li><b>Width</b> every 2 px along each line; 60 px at each tip left out. ' + (
                'Read from the <b>segmentation mask</b> (trained SegFormer, run on full-resolution tiles): across the line, the run of '
                'glue pixels, with its two ends placed between pixels where the probability passes one half.'
                if mask_model is not None else
                'Between the two edges of the glue band, placed by the trained trees.') + '</li>'
            '<li><b>Millimetres</b>: width in pixels divided by the fitted pixels-per-mm at that place.</li>'
            "<li><b>Departure</b>: width minus the line's own gradual course (a smooth curve from end to end; a gradual change along a whole line is mostly the uncalibrated camera and is shown separately).</li>"
            f'<li><b>Test</b>: the departure is averaged over windows of {", ".join(f"{size * STEP:.0f}" for size in SIZES)} px. '
            'The limits are taken from good lines themselves: set so that 95 % of good lines stay under them at every window length '
            'together (&ldquo;significant&rdquo;), or 80 % (&ldquo;notable&rdquo;). No normal distribution is assumed: neighbouring widths are '
            'correlated and the spread has heavy tails, so 3-sigma limits would flag far too much.</li>'
            '<li><b>Segmentation glitches.</b> The mask itself is checked along every line: where it is <i>missing</i>, where its edge '
            '<i>fades out</i> instead of ending abruptly (the model is unsure there; the softest 2 % of good readings), and where its middle '
            '<i>steps sideways</i> beyond what good lines show. Every width change found is then judged: <b>glitch</b> if the mask is in '
            'question over 30 % of it; <b>also seen in the band edges</b> if the width read straight from the picture (band edges, not the '
            'mask) changes the same way by at least half as much; otherwise <b>in the mask only</b>.</li>'
            '<li>Limits for a frame are computed from other frames (five groups by capture time)'
            + (f', and only from the {sum(frame["fresh"] for frame in frames)} frames the mask model was not trained on '
               '(marked &ldquo;new to the model&rdquo; below): a model reads frames it was trained on more evenly.'
               if held is not None else '.') + '</li></ol>',
            '<p><b>Reading the results.</b> The supplied strips have no defects, so what is marked here is the largest natural variation of good '
            f'lines, not faults: by construction about 5 % of good lines are marked significant and 20 % notable or more. Found: '
            f'{counts["significant"]} of {len(every)} lines with a significant place, {counts["notable"]} with a notable one'
            + (f' (among the {sum(line["fresh"] for line in every)} lines new to the model: '
               f'{sum(line["fresh"] and any(f["level"] == "significant" for f in line["findings"]) for line in every)} and '
               f'{sum(line["fresh"] and any(f["level"] == "notable" for f in line["findings"]) for line in every)})'
               if held is not None else '') + '. '
            'The limits in the table are the smallest change this method can tell from a good line, by length of the change.</p>',
            '<p><b>Glitches.</b> ' + verdict_text + '</p>',
            f'<table><tr><th rowspan="2">Fabric</th><th rowspan="2">Lines</th><th rowspan="2">Pixels per mm</th>'
            f'<th rowspan="2">Line width, mm (mean &plusmn; SD between lines)</th><th rowspan="2">Spread along a line, mm (40 px blocks)</th>'
            f'<th colspan="{len(SIZES)}">Smallest change marked significant, mm, when it lasts</th><th rowspan="2">Lines with a significant place</th></tr>'
            f'<tr>{lengths}</tr>' + ''.join(rows) + '</table>',
            '<p>Pictures: the band is tinted <span class="w">red where wider</span> and <span class="n">blue where narrower</span> than the '
            f"line's own median (full colour at &plusmn;{SHOW_MM} mm). Marked places are outlined (thick = significant, thin = notable) with the "
            'change in mm. Charts: width along the line; green band = normal range of 40 px averages around the line&rsquo;s gradual course '
            '(dashed); grey line = 4 mm.</p>']
    for frame in frames:
        if not frame['lines']:
            continue
        picture = draw(cv2.imread(frame['image']), frame['lines'])
        name = f"{frame['colour']}_{frame['frame']}.jpg"
        cv2.imwrite(str(out / 'frames' / name), cv2.resize(picture, (2304, 1728), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        tag = '' if held is None else (' &nbsp; <small>(new to the model)</small>' if frame['fresh']
                                        else ' &nbsp; <small>(used to train the mask model)</small>')
        page.append(f'<div class="f"><h2>{frame["colour"]} &nbsp; {frame["frame"]}{tag}</h2>'
                    f'<a href="frames/{name}"><img loading="lazy" src="frames/{name}" width="1150"></a>')
        for line in frame['lines']:
            fold = fold_of[line['frame']]
            own = limits[fold][line['colour']]
            limits_mm = {level: {size: own[level][size] * spread[line['colour']] for size in SIZES} for level in own}
            marks = []
            for finding in line['findings']:
                long_mm = (finding['last'] - finding['first'] + 1) * STEP / float(np.mean(line['per_mm']))
                at_mm = finding['first'] * STEP / float(np.mean(line['per_mm']))
                kind = 'w' if finding['change'] > 0 else 'n'
                strong = ' s' if finding['level'] == 'significant' else ''
                marks.append(f'<span class="{kind}{strong}">{"wider" if finding["change"] > 0 else "narrower"} by '
                             f'{abs(finding["change"]):.2f} mm over {long_mm:.0f} mm, from {at_mm:.0f} mm ({finding["level"]}; '
                             f'{VERDICT[finding["verdict"]]})</span>')
            scale_here = float(np.mean(line['per_mm']))
            faults = '; '.join(f'{reason} for {(last - first + 1) * STEP / scale_here:.0f} mm at {first * STEP / scale_here:.0f} mm'
                               for first, last, reason in line['glitches'])
            whole = f' &nbsp; <b>whole line {line["whole"]}ly off 4 mm</b>' if line['whole'] else ''
            page.append(f'<p><b>Line {line["number"]}</b>: median {line["median_mm"]:.2f} mm '
                        f'({np.median(line["width_px"]):.1f} px at {np.mean(line["per_mm"]):.2f} px/mm), '
                        f'range {np.percentile(line["mm"], 1):.2f}-{np.percentile(line["mm"], 99):.2f} mm, '
                        f'gradual change along the line {np.ptp(line["trend"]):.2f} mm{whole}<br>'
                        + (html.unescape('; '.join(marks)) if marks else 'no place beyond the normal variation of good lines')
                        + (f'<br><span class="g">Segmentation glitches: {faults}</span>' if faults else '')
                        + f'</p>{chart(line, limits_mm, float(np.mean(line["per_mm"])))}')
        page.append('</div>')
    (out / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(f'{len(every)} lines; scale {mean_per_mm:.2f} px/mm; spread {spread}; '
          f'significant on {counts["significant"]} lines, notable on {counts["notable"]}')
    print(verdict_text)
    print('limits (significant), mm:', {colour: {int(size * STEP): round(all_limits[colour]['significant'][size] * spread[colour], 3)
                                                  for size in SIZES} for colour in colours})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
