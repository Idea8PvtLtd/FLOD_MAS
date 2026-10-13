"""Make the example pictures of the handbook (``figures/*.jpg``).

    python docs/handbook/pictures.py

Unlike the diagrams and charts, these come from files that are **not** in the repository:
the sample frames (``enhancement_images/``, git-ignored) and the result pages and cached
runs of the experiments (``~/flod_experiments/``). The pictures themselves are committed, so
the handbook can be rebuilt without them; run this only to remake a picture.
"""
import pickle
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENTS = Path.home() / 'flod_experiments'
PAGES = EXPERIMENTS / 'out'
FIGURES = HERE / 'figures'
FONT = cv2.FONT_HERSHEY_SIMPLEX


def save(name, image, width=1500, quality=82):
    if image.shape[1] > width:
        height = round(image.shape[0] * width / image.shape[1])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FIGURES / name), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(name, image.shape[1], 'x', image.shape[0])


def headed(image, labels, band=64, scale=1.0):
    """The picture with a white band on top naming its columns."""
    width = image.shape[1]
    head = np.full((band, width, 3), 255, np.uint8)
    step = width / len(labels)
    for number, label in enumerate(labels):
        size = cv2.getTextSize(label, FONT, scale, 2)[0]
        cv2.putText(head, label, (int(number * step + (step - size[0]) / 2), band - 20), FONT, scale,
                    (40, 40, 40), 2, cv2.LINE_AA)
        if number:
            cv2.line(head, (int(number * step), 8), (int(number * step), band - 8), (190, 190, 190), 2)
    return np.vstack([head, image])


def sided(image, labels, band=150, scale=1.0, top=0):
    """The picture with a white band on the left naming its rows (below the first ``top`` pixels)."""
    height = image.shape[0]
    side = np.full((height, band, 3), 255, np.uint8)
    step = (height - top) / len(labels)
    for number, label in enumerate(labels):
        cv2.putText(side, label, (12, int(top + number * step + step / 2 + 10)), FONT, scale, (40, 40, 40), 2,
                    cv2.LINE_AA)
    return np.hstack([side, image])


def object_on_three_colours():
    """Plain picture and contrast view of the middle of one line per colour."""
    rows = []
    for name in ('dark_230044_057574_line2', 'pink_230210_313721_line1', 'white_230306_169300_line2'):
        sheet = cv2.imread(str(PAGES / 'compare_stages' / 'img' / f'{name}.jpg'))
        cell = sheet.shape[0] // 3
        rows.append(sheet[cell:2 * cell, :2 * cell])
    picture = headed(np.vstack(rows), ['as photographed', 'contrast raised'], scale=1.0)
    save('photo_object.jpg', sided(picture, ['dark', 'pink', 'white'], scale=1.3, top=64), width=1100)


def stages_of_one_frame(frame='230210_313721'):
    """One frame: as taken, the region model's panels, the rough lines, the final ribbons."""
    run = pickle.load(open(EXPERIMENTS / 'segformer' / 'runs' / f'{frame}.pkl', 'rb'))
    image = cv2.imread(str(REPO / run['image']))
    region = np.unpackbits(run['roi'])[:run['shape'][0] * run['shape'][1]].reshape(run['shape'])
    with_region = image.copy()
    with_region[region > 0] = (0.55 * with_region[region > 0] + 0.45 * np.array([60, 60, 235])).astype(np.uint8)
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(with_region, contours, -1, (0, 0, 255), 10)
    with_rough = image.copy()
    for line in run['rough']:
        cv2.polylines(with_rough, [np.round(line['ring']).astype(np.int32)], True, (0, 220, 255), 12)
    with_final = image.copy()
    for line in run['lines']:
        cv2.polylines(with_final, [np.round(line['ring']).astype(np.int32)], True, (0, 235, 0), 12)
    panels = [image, with_region, with_rough, with_final]
    names = ['1  frame as taken (4608 x 3456)', '2  panel regions from the model',
             '3  rough lines (local detector)', '4  final ribbons']
    small = []
    for panel, name in zip(panels, names):
        panel = cv2.resize(panel, (1100, 825), interpolation=cv2.INTER_AREA)
        small.append(headed(panel, [name], band=58, scale=0.95))
    gap = np.full((small[0].shape[0], 12, 3), 255, np.uint8)
    rows = [np.hstack([small[0], gap, small[1]]), np.hstack([small[2], gap, small[3]])]
    save('photo_stages_frame.jpg', np.vstack([rows[0], np.full((12, rows[0].shape[1], 3), 255, np.uint8), rows[1]]),
         width=1600)


def versions_close_up():
    sheet = cv2.imread(str(PAGES / 'compare_stages' / 'img' / 'white_230306_169300_line2.jpg'))
    labels = ['as photographed', 'contrast view', 'version 2', 'version 3', 'version 4', 'version 5']
    save('photo_versions.jpg', headed(sheet, labels, scale=1.5, band=84), width=1600)


def states_along_a_line():
    sheet = cv2.imread(str(PAGES / 'v4_m1_all' / 'crops' / 'white_230305_274282.jpg'))
    labels = ['start', 'start: result', 'middle', 'middle: result', 'end', 'end: result']
    save('photo_states.jpg', headed(sheet, labels, scale=1.6, band=90), width=1600)


def width_report_frame():
    save('photo_width_report.jpg', cv2.imread(str(PAGES / 'width_report' / 'frames' / 'dark_230007_642490.jpg')),
         width=1400)


def main_model():
    sheet = cv2.imread(str(PAGES / 'main_model_check' / 'img' / '0041_frame_line1.jpg'))
    labels = ['as photographed', 'contrast view', 'main model, as shipped', 'main + trees + ribbon',
              'version 5']
    save('photo_main_model.jpg', headed(sheet, labels, scale=1.2, band=80), width=1500)


def trained_segmenter():
    sheet = cv2.imread(str(PAGES / 'segformer_vs_sam' / 'img' / '0041_frame_line1.jpg'))
    labels = ['as photographed', 'contrast view', 'version 5 (SAM 2.1)', 'SegFormer mask',
              'SegFormer + ribbon']
    save('photo_segformer.jpg', headed(sheet, labels, scale=1.2, band=80), width=1500)


def tool_window():
    save('photo_lab_window.jpg', cv2.imread(str(PAGES / 'lab_window3.png')), width=1500, quality=85)


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    for make in (object_on_three_colours, stages_of_one_frame, versions_close_up, states_along_a_line,
                 width_report_frame, main_model, trained_segmenter, tool_window):
        make()
