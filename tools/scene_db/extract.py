"""Frames of the recordings as pictures: python -m tools.scene_db.extract [video ...] [--key]

Without --key: every frame as data/scene_db/frames/<video>/<frame:04d>.jpg, 640 px
wide, JPEG quality 85. These are what the contact sheets and the review use.

With --key: the middle frame of every still segment (stage bare_bed, board_empty
or board_cloth in segments.csv) as a full-resolution PNG,
data/scene_db/key/<video>_<frame:04d>.png. Labels must exist first.

Existing still photos are not copied; stills.csv names them by path.
"""
import argparse
from time import perf_counter
import cv2
from . import DATABASE, all_videos, find_video, read_segments, relative, video_frames

FRAME_WIDTH = 640
STILL_STAGES = ('bare_bed', 'board_empty', 'board_cloth')


def key_frames(video):
    """Frame numbers of the middle of each still segment of this video."""
    segments = read_segments(relative(video))
    return [(segment['start_frame'] + segment['end_frame']) // 2
            for segment in segments if segment['stage'] in STILL_STAGES]


def extract(video, key=False):
    name = video.stem
    wanted = set(key_frames(video)) if key else None
    folder = DATABASE / ('key' if key else 'frames' + '/' + name)
    folder.mkdir(parents=True, exist_ok=True)
    started, written = perf_counter(), 0
    for index, _, frame in video_frames(video):
        if key:
            if index in wanted:
                cv2.imwrite(str(folder / f'{name}_{index:04d}.png'), frame)
                written += 1
            continue
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (FRAME_WIDTH, round(height * FRAME_WIDTH / width)),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f'{index:04d}.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written += 1
    print(f'{name}: {written} {"key frames" if key else "frames"} in '
          f'{perf_counter() - started:.1f} s -> {folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('videos', nargs='*', help='video paths or time stamps; default: all')
    parser.add_argument('--key', action='store_true', help='full-resolution key frames from the labels')
    arguments = parser.parse_args()
    for video in ([find_video(name) for name in arguments.videos] or all_videos()):
        extract(video, arguments.key)


if __name__ == '__main__':
    main()
