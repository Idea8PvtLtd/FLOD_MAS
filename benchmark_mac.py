"""MAC ONLY: time the original models against their Core ML packages on real frames.

    python benchmark_mac.py                     # every video in video_testcases/
    python benchmark_mac.py --every 50 --limit 12

Frames are taken from the test recordings and rotated as the camera would be
(config.json). Each frame goes through the fabric gate and the full local
inspection (source mask + SegFormer per line) twice: once with the original
ONNX/H5 files on the CPU, once with Core ML. Timings and the agreement of the
two results are printed; side-by-side overlays are written to
benchmark_results/ to compare by eye. Lens undistortion is skipped, so this
needs no calibration file; pixel measurements are compared, not millimetres.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def frames(videos, every, limit):
    rotation = ROTATE.get(int(CONFIG['cameras'][0].get('rotation', 0)))
    taken = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        index = 0
        while taken < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                taken += 1
                yield f'{video.stem}_f{index:05d}', cv2.rotate(frame, rotation) if rotation is not None else frame
            index += 1
        capture.release()


def build(coreml):
    """Gate and inspector with Core ML on or off; the switch is read at load time."""
    os.environ['FLOD_COREML'] = '1' if coreml else '0'
    from auto_trigger.models import FabricGate
    from inspection.local.engine import LiveInspection
    local = CONFIG['local_inspection']
    started = perf_counter()
    gate = FabricGate(CONFIG['auto_trigger'])
    inspector = LiveInspection(ROOT / local['source_model'], ROOT / local['glue_model'],
                               local['offset_pixels'], local['source_width'],
                               CONFIG['inspection'].get('segment_count', 10))
    return gate, inspector, perf_counter() - started


def timed(function, *args):
    started = perf_counter()
    value = function(*args)
    return value, (perf_counter() - started) * 1000


def inspect(gate, inspector, frame):
    """One frame end to end, with each model's share of the time."""
    verdict, fabric_ms = timed(gate.classify, frame)
    mask, source_ms = timed(inspector.source.predict, frame)
    glue = inspector.glue
    calls = []
    original_predict = glue.predict
    def counted(image):
        value, ms = timed(original_predict, image)
        calls.append(ms)
        return value
    glue.predict = counted
    try:
        lines, inspect_ms = timed(lambda: list(inspector.run_lines(frame, on_error=lambda *_: None)), )
        error = None
    except ValueError as exc:
        lines, inspect_ms, error = [], 0.0, str(exc)
    finally:
        glue.predict = original_predict
    return dict(verdict=verdict, fabric_ms=fabric_ms, source_mask=mask, source_ms=source_ms,
                glue_ms=calls, inspect_ms=inspect_ms, lines=lines, error=error)


def summary(name, values):
    values = [v for v in values if v is not None]
    if not values:
        return f'{name:34s} -'
    return f'{name:34s} median {np.median(values):8.1f} ms   mean {np.mean(values):8.1f} ms   (n={len(values)})'


def panel(frame, result, title):
    image = result['lines'][-1]['overlay'] if result['lines'] else frame.copy()
    if not result['lines']:
        tint = image.copy()
        tint[result['source_mask'] > 0] = (0, 200, 0)
        image = cv2.addWeighted(image, .6, tint, .4, 0)
    scale = 900 / image.shape[1]
    image = cv2.resize(image, (900, round(image.shape[0] * scale)))
    total = result['fabric_ms'] + result['source_ms'] + result['inspect_ms']
    label = f"{title}: {total:.0f} ms  lines {len(result['lines'])}  fabric {result['verdict']['label']}"
    cv2.rectangle(image, (0, 0), (900, 44), (20, 20, 20), -1)
    cv2.putText(image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', type=Path, default=ROOT / 'video_testcases')
    parser.add_argument('--every', type=int, default=60, help='take one frame in this many')
    parser.add_argument('--limit', type=int, default=20, help='frames in total')
    parser.add_argument('--out', type=Path, default=ROOT / 'benchmark_results')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    cv2.setNumThreads(CONFIG.get('performance', {}).get('opencv_threads', 1))

    cpu = build(coreml=False)
    mac = build(coreml=True)
    print(f'model loading: original {cpu[2]:.2f}s, Core ML {mac[2]:.2f}s')
    backends = {'original': cpu, 'coreml': mac}
    for gate, inspector, _ in backends.values():  # first calls compile; keep them out of the timings
        warm = np.zeros((int(CONFIG['cameras'][0]['height']), int(CONFIG['cameras'][0]['width']), 3), np.uint8)
        gate.classify(warm)
        inspector.source.predict(warm)
        inspector.glue.predict(np.zeros((64, 1024, 3), np.uint8))

    times = {name: {'fabric': [], 'source': [], 'glue': [], 'inspect': [], 'total': []} for name in backends}
    agreement, gate_same, width_diff, line_mismatch = [], 0, [], 0
    count = 0
    for name, frame in frames(sorted(args.videos.glob('*.mp4')), args.every, args.limit):
        results = {key: inspect(gate, inspector, frame) for key, (gate, inspector, _) in backends.items()}
        for key, result in results.items():
            t = times[key]
            t['fabric'].append(result['fabric_ms']); t['source'].append(result['source_ms'])
            t['glue'].extend(result['glue_ms']); t['inspect'].append(result['inspect_ms'] or None)
            t['total'].append(result['fabric_ms'] + result['source_ms'] + result['inspect_ms'])
        a, b = results['original'], results['coreml']
        same = float((a['source_mask'] == b['source_mask']).mean())
        agreement.append(same)
        gate_same += a['verdict']['label'] == b['verdict']['label']
        if len(a['lines']) != len(b['lines']):
            line_mismatch += 1
        else:
            for line_a, line_b in zip(a['lines'], b['lines']):
                width_diff.append(abs(line_a['mean_width_px'] - line_b['mean_width_px']))
        count += 1
        print(f"{name}: source mask agreement {same:.4%}  lines {len(a['lines'])}/{len(b['lines'])}  "
              f"fabric {a['verdict']['label']}/{b['verdict']['label']}  "
              f"total {times['original']['total'][-1]:.0f} ms -> {times['coreml']['total'][-1]:.0f} ms", flush=True)
        cv2.imwrite(str(args.out / f'{name}.jpg'),
                    np.hstack([panel(frame, a, 'ORIGINAL (CPU)'), panel(frame, b, 'CORE ML')]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    print()
    for stage in ('fabric', 'source', 'glue', 'inspect', 'total'):
        label = {'fabric': 'fabric classifier', 'source': 'source mask (UNet + pre/post)',
                 'glue': 'SegFormer per line', 'inspect': 'full line inspection',
                 'total': 'frame total'}[stage]
        print(summary(f'original  {label}', times['original'][stage]))
        print(summary(f'Core ML   {label}', times['coreml'][stage]))
    print(f'\nframes {count}; fabric verdict identical on {gate_same}/{count}; '
          f'source mask pixel agreement min {min(agreement):.4%} mean {np.mean(agreement):.4%}')
    print(f'frames with a different number of lines: {line_mismatch}/{count}')
    if width_diff:
        print(f'mean line width difference where lines match: max {max(width_diff):.3f} px, '
              f'mean {np.mean(width_diff):.3f} px')
    print(f'side-by-side overlays: {args.out}')


if __name__ == '__main__':
    main()
