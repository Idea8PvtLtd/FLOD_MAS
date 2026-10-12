"""How much the readings change when the same photo is moved by one pixel.

Run: python tests/verify_shift_repeatability.py IMAGE [IMAGE ...] [--smooth-normal-rows N] [--smooth-logits]

A good measurement should hardly notice a one-pixel move of the picture. This runs the
real local models on each photo and on copies moved by one pixel, and prints how far the
line means and the ten segment averages differ. Without options it uses the settings in
config.json; the options override them so two settings can be compared on the same photos.
Does not start a camera, call a network service or save anything.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspection.local.engine import LiveInspection

SHIFTS = ((1, 0), (0, 1))          # (right, down) in pixels


def differences(inspector, frame):
    """(line mean differences, segment average differences) in pixels for one photo."""
    def lines(image):
        return list(inspector.run_lines(image, on_error=lambda index, error: None))
    reference = lines(frame)
    line_changes, segment_changes = [], []
    for right, down in SHIFTS:
        moved = lines(np.roll(frame, (down, right), axis=(0, 1)))
        if len(moved) != len(reference):
            raise ValueError(f'The number of lines changed from {len(reference)} to {len(moved)} '
                             f'when the photo moved by ({right}, {down})')
        for first, second in zip(reference, moved):
            line_changes.append(abs(first['mean_width_px'] - second['mean_width_px']))
            segment_changes += [abs(a['average_width_px'] - b['average_width_px'])
                                for a, b in zip(first['segments'], second['segments'])
                                if a['average_width_px'] is not None and b['average_width_px'] is not None]
    return line_changes, segment_changes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--smooth-normal-rows', type=int)
    parser.add_argument('--smooth-logits', action='store_true', default=None)
    args = parser.parse_args()
    config = json.loads((ROOT/'config.json').read_text())['local_inspection']
    rows = config.get('smooth_normal_rows', 0) if args.smooth_normal_rows is None else args.smooth_normal_rows
    logits = config.get('smooth_logits', False) if args.smooth_logits is None else args.smooth_logits
    inspector = LiveInspection(ROOT/config['source_model'], ROOT/config['glue_model'],
                               config['offset_pixels'], config['source_width'],
                               smooth_normal_rows=rows, smooth_logits=logits)
    print(f'smooth_normal_rows={rows} smooth_logits={logits}', flush=True)
    line_changes, segment_changes = [], []
    for path in args.images:
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        lines, segments = differences(inspector, frame)
        print(f'{path.name}: line mean max {max(lines):.2f} px, segment average max {max(segments):.2f} px', flush=True)
        line_changes += lines
        segment_changes += segments
    print(json.dumps({
        'photos': len(args.images), 'line_pairs': len(line_changes), 'segment_pairs': len(segment_changes),
        'line_mean_change_px': {'mean': round(float(np.mean(line_changes)), 3), 'max': round(float(np.max(line_changes)), 3)},
        'segment_average_change_px': {'mean': round(float(np.mean(segment_changes)), 3),
                                      '95th': round(float(np.percentile(segment_changes, 95)), 3),
                                      'max': round(float(np.max(segment_changes)), 3)}}, indent=2))


if __name__ == '__main__':
    main()
