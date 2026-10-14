"""Signals of every frame of a recording: python -m tools.scene_db.signals_trace [video ...]

Writes tests/fixtures/scene/trace_<time stamp>.csv, one row per frame:

  frame        0-based frame number
  time_s       frame / fps
  motion       share (0..1) of the small picture that differs from the frame before
               (auto_trigger.scene_monitor.scene_signals)
  markers_all  ArUco markers seen on a 1280 wide copy
  markers_id0  of those, the ID-0 markers (2 or more: the bed is bare)
  hand         1 when the hand model (auto_trigger.models.HandPresence, built from
               config.json as the application builds it) reports a hand, else 0.
               The model runs over the frames in order, as it does live.
  thumb        the small picture averaged down to 24 x 18 grey cells, base64. Replay
               from a trace uses it for the "changed since the key photo" test.

--no-hand leaves the hand column empty (no MediaPipe needed).
--dump-small <file.npy> also stores the full small pictures (for experiments).
"""
import argparse
from time import perf_counter
import numpy as np
from . import (BARE_MARKERS, TRACE_COLUMNS, all_videos, count_markers, encode_thumb,
               find_video, load_config, marker_detector, signal_function, thumbnail,
               trace_path, video_frames, write_rows)


def frame_signals(video, with_hand=True, on_frame=None):
    """Yield one trace row (dict) per frame; `small` rides along under '_small'."""
    signals, _ = signal_function()
    detector = marker_detector()
    hand = None
    if with_hand:
        from auto_trigger.models import HandPresence
        hand = HandPresence(load_config()['auto_trigger'])
    previous = None
    try:
        for index, fps, frame in video_frames(video):
            small, motion = signals(frame, previous)
            previous = small
            markers_all, markers_id0 = count_markers(detector, frame)
            row = {'frame': index, 'time_s': f'{index / fps:.2f}', 'motion': f'{motion:.5f}',
                   'markers_all': markers_all, 'markers_id0': markers_id0,
                   'hand': '' if hand is None else int(hand.present(frame)),
                   'thumb': encode_thumb(thumbnail(small)), '_small': small}
            if on_frame is not None:
                on_frame(index, frame, row)
            yield row
    finally:
        if hand is not None:
            hand.close()


def trace(video, with_hand=True, dump_small=None):
    started = perf_counter()
    rows = list(frame_signals(video, with_hand))
    seconds = perf_counter() - started
    target = trace_path(video)
    write_rows(target, TRACE_COLUMNS, rows)
    if dump_small:
        np.save(dump_small, np.stack([row['_small'] for row in rows]))
    bare = sum(int(row['markers_id0']) >= BARE_MARKERS for row in rows)
    hands = sum(row['hand'] == 1 for row in rows)
    print(f'{video.name}: {len(rows)} frames in {seconds:.1f} s '
          f'({seconds / max(1, len(rows)) * 1000:.0f} ms a frame); bare bed in {bare}, '
          f'hand in {hands if with_hand else "-"} -> {target}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('videos', nargs='*', help='video paths or time stamps; default: all')
    parser.add_argument('--no-hand', action='store_true', help='leave the hand column empty')
    parser.add_argument('--dump-small', help='also save the small pictures to this .npy file '
                        '(one video only)')
    arguments = parser.parse_args()
    print('motion from', signal_function()[1])
    for video in ([find_video(name) for name in arguments.videos] or all_videos()):
        trace(video, not arguments.no_hand, arguments.dump_small)


if __name__ == '__main__':
    main()
