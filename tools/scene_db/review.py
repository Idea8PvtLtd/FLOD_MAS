"""Contact sheets for checking the labels by eye: python -m tools.scene_db.review [--videos] [--stills]

Into data/scene_db/review/:

  <video>_<start>-<end>.jpg   one sheet per 10 s of a recording, every 5th frame, each
                              stamped with frame number, time, the stage from
                              segments.csv and the signals of the trace
                              (m = motion %, id0 = ID-0 markers, H = hand model says hand)
  stills_<stage>_<n>.jpg      the still photos of stills.csv, 30 to a sheet, by stage,
                              each stamped with its number in stills.csv, folder and file
                              name, cloth count, colour, orientation and hand flag

--step and --span change the sampling of the video sheets (e.g. --step 1 --span 40
--from 270 --to 320 for a close look at a boundary).
The frame pictures come from data/scene_db/frames (run extract first).
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from . import (DATABASE, ROOT, STILLS, all_videos, find_video, read_rows, read_segments,
               read_trace, relative, segment_of, trace_path)

REVIEW = DATABASE / 'review'
STAGE_COLOURS = {'bare_bed': (160, 160, 160), 'board_empty': (0, 200, 255),
                 'board_cloth': (0, 220, 0), 'hand': (0, 0, 255), 'moving': (255, 120, 0)}


def stamp(picture, lines, colour=(255, 255, 255)):
    """Text lines on a dark band at the top of a thumbnail."""
    scale = picture.shape[1] / 400
    height = int(20 * scale)
    cv2.rectangle(picture, (0, 0), (picture.shape[1], height * len(lines) + 4), (0, 0, 0), -1)
    for row, text in enumerate(lines):
        cv2.putText(picture, text, (4, height * (row + 1) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, colour if row == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return picture


def sheet(thumbnails, columns):
    """Thumbnails of one size laid out in rows."""
    blank = np.zeros_like(thumbnails[0])
    rows = []
    for start in range(0, len(thumbnails), columns):
        row = thumbnails[start:start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    return np.vstack(rows)


def video_sheets(video, step=5, span=100, first=0, last=None, width=400, columns=5):
    name = video.stem
    frames = DATABASE / 'frames' / name
    trace = read_trace(trace_path(video)) if trace_path(video).is_file() else []
    try:
        segments = read_segments(relative(video))
    except FileNotFoundError:
        segments = []
    count = len(list(frames.glob('*.jpg')))
    if not count:
        raise SystemExit(f'No frames in {frames}: run python -m tools.scene_db.extract first')
    last = count - 1 if last is None else min(last, count - 1)
    written = []
    for start in range(first, last + 1, span):
        thumbnails = []
        end = min(start + span - 1, last)
        for index in range(start, end + 1, step):
            picture = cv2.imread(str(frames / f'{index:04d}.jpg'))
            picture = cv2.resize(picture, (width, round(picture.shape[0] * width / picture.shape[1])))
            segment = segment_of(segments, index)
            stage = segment['stage'] if segment else '?'
            first_line = f'{index} {stage}'
            if segment and segment['stage'] == 'board_cloth':
                first_line += f' x{segment["cloth_count"]}'
            second = ''
            if index < len(trace):
                row = trace[index]
                second = (f'{row["time_s"]:.1f}s m{row["motion"] * 100:.1f} id0={row["markers_id0"]}'
                          + (' H' if row['hand'] else ''))
            thumbnails.append(stamp(picture, [first_line, second],
                                    STAGE_COLOURS.get(stage, (255, 255, 255))))
        target = REVIEW / f'{name}_{start:04d}-{end:04d}{"" if step == 5 else f"_s{step}"}.jpg'
        REVIEW.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), sheet(thumbnails, columns), [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(target)
    return written


def still_thumbnail(path, width):
    """A still as a thumbnail; JPEGs are decoded at an eighth of their size, which is fast."""
    picture = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8 if Path(path).suffix.lower()
                         in ('.jpg', '.jpeg') else cv2.IMREAD_COLOR)
    if picture is None:
        picture = np.zeros((3, 4, 3), np.uint8)
    # One cell size for every photo, whatever its shape.
    scale = min(width / picture.shape[1], width * 0.75 / picture.shape[0])
    picture = cv2.resize(picture, (max(1, round(picture.shape[1] * scale)),
                                   max(1, round(picture.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    cell = np.zeros((round(width * 0.75), width, 3), np.uint8)
    cell[:picture.shape[0], :picture.shape[1]] = picture
    return cell


def still_sheets(per_sheet=30, width=400, columns=6, only=None):
    rows = read_rows(STILLS)
    for number, row in enumerate(rows):
        row['number'] = number
    written = []
    REVIEW.mkdir(parents=True, exist_ok=True)
    for stage in sorted({row['stage'] for row in rows}):
        if only and stage != only:
            continue
        chosen = [row for row in rows if row['stage'] == stage]
        for part, start in enumerate(range(0, len(chosen), per_sheet)):
            thumbnails = []
            for row in chosen[start:start + per_sheet]:
                path = Path(row['path'])
                name = path.name if len(path.name) < 34 else path.name[:22]
                details = (f'x{row["cloth_count"]} {row["colour"]} {row["orientation"]}'
                           + (' HAND' if row['has_hand'] == '1' else '')
                           + (' ok' if row['reviewed'] == '1' else ''))
                thumbnails.append(stamp(still_thumbnail(ROOT / path, width),
                                        [f'#{row["number"]} {path.parent.name[:14]}', name, details],
                                        STAGE_COLOURS.get(stage, (255, 255, 255))))
            target = REVIEW / f'stills_{stage}_{part:02d}.jpg'
            cv2.imwrite(str(target), sheet(thumbnails, columns), [cv2.IMWRITE_JPEG_QUALITY, 85])
            written.append(target)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--videos', nargs='*', help='video sheets (default: all videos)')
    parser.add_argument('--stills', action='store_true', help='still sheets')
    parser.add_argument('--stage', help='still sheets of this stage only')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--span', type=int, default=100)
    parser.add_argument('--from', dest='first', type=int, default=0)
    parser.add_argument('--to', dest='last', type=int)
    parser.add_argument('--width', type=int, default=400)
    arguments = parser.parse_args()
    both = arguments.videos is None and not arguments.stills
    written = []
    if arguments.videos is not None or both:
        for video in ([find_video(name) for name in arguments.videos or ()] or all_videos()):
            written += video_sheets(video, arguments.step, arguments.span, arguments.first,
                                    arguments.last, arguments.width)
    if arguments.stills or both:
        written += still_sheets(only=arguments.stage, width=arguments.width)
    for target in written:
        print(target)


if __name__ == '__main__':
    main()
