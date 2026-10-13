"""Draw the data charts of the handbook (``figures/chart_*.svg``).

Every number here is copied from a table of ``docs/glue-line/GLUE_LINE_FINDINGS.md``; the
section is named beside it. Change the number there first, then here.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent / 'figures'
COLOURS = ('dark', 'pink', 'white')
FABRIC = {'dark': '#3d4a5c', 'pink': '#e59ab0', 'white': '#b9bfc8'}
STAGE = ('#7c93b5', '#e0a458', '#c8553d', '#b8bec8')

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e3e7ee',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'svg.fonttype': 'none',
                     'legend.frameon': False})


def save(figure, name):
    figure.tight_layout()
    figure.savefig(FIGURES / f'chart_{name}.svg')
    plt.close(figure)


def time_per_stage():
    """*Time per frame* and *Local GPU flow, Where the time went*: seconds per frame."""
    stages = ('ROI model', 'Rough line', 'SAM', 'Everything else')
    cloud = {'dark': (2.1, 10.6, 8.8, 0.7), 'pink': (2.1, 8.7, 33.0, 1.1), 'white': (2.4, 8.1, 11.8, 0.7)}
    local = {'dark': (0.25, 0.94, 3.85, 0.7), 'pink': (0.24, 0.93, 4.57, 0.8), 'white': (0.22, 0.71, 3.21, 0.6)}
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    names, rows = [], []
    for colour in COLOURS:
        names += [f'{colour}, cloud (version 4)', f'{colour}, local GPU (version 5)']
        rows += [cloud[colour], local[colour]]
    left = [0.0] * len(rows)
    for index, stage in enumerate(stages):
        values = [row[index] for row in rows]
        axis.barh(names, values, left=left, color=STAGE[index], label=stage, height=0.62)
        left = [a + b for a, b in zip(left, values)]
    for name, total in zip(names, left):
        axis.text(total + 0.5, name, f'{total:.1f} s', va='center', fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel('seconds per frame (4608 x 3456)')
    axis.set_xlim(0, 50)
    axis.legend(ncol=4, loc='lower right', fontsize=8)
    save(figure, 'time_per_stage')


def time_two_pipelines():
    """*The main branch's local model, Time, stage by stage*: seconds per frame, same nine frames."""
    flows = (('Version 5 (GPU)', (('ROI model', 0.19, '#7c93b5'), ('Rough line', 0.63, '#e0a458'),
                                  ('SAM 2.1', 3.13, '#c8553d'), ('Repair, edges, ribbon, widths', 0.22, '#2f6f4f'),
                                  ('Other', 0.12, '#b8bec8'))),
             ('Pipeline on main (CPU)', (('Strip U-Net', 0.91, '#7c93b5'), ('Source lines', 0.32, '#e0a458'),
                                         ('CLAHE, per line', 1.08, '#c8553d'), ('SegFormer-B0', 0.09, '#7b5ea7'),
                                         ('Unfold, refold', 0.10, '#d9c9ee'), ('Clean, width on rays', 0.62, '#2f6f4f'),
                                         ('Other', 0.30, '#b8bec8'))))
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True)
    for axis, (name, stages) in zip(axes, flows):
        left = 0.0
        for stage, seconds, colour in stages:
            axis.barh([name], [seconds], left=left, color=colour, label=f'{stage} {seconds:.2f}', height=0.5)
            left += seconds
        axis.text(left + 0.06, 0, f'{left:.1f} s', va='center', fontsize=8)
        axis.set_xlim(0, 5.4)
        axis.legend(ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.75), columnspacing=1.0)
    axes[1].set_xlabel('seconds per frame (4608 x 3456, 2-4 lines)')
    save(figure, 'time_two_pipelines')


def edge_error():
    """*Accuracy, Candidate edge methods*, held-out frames: edge error, px RMS."""
    methods = ('Steepest slope\n(version 4)', 'Profile fit', 'Trained trees\n(version 5)')
    cases = {'Standard, dark': (0.98, 0.69, 0.63), 'Standard, pink': (1.02, 0.83, 0.63),
             'Core along one edge, dark': (1.87, 1.65, 0.79), 'Core along one edge, pink': (3.71, 3.46, 1.13),
             'Blurred 4 px, dark': (1.00, 1.15, 0.54)}
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    width = 0.26
    for index, method in enumerate(methods):
        places = [number + (index - 1) * width for number in range(len(cases))]
        values = [case[index] for case in cases.values()]
        axis.bar(places, values, width, label=method.replace('\n', ' '), color=('#b8bec8', '#7c93b5', '#2f6f4f')[index])
        for place, value in zip(places, values):
            axis.text(place, value + 0.05, f'{value:.2f}', ha='center', fontsize=7)
    axis.set_xticks(range(len(cases)), [name.replace(', ', ',\n') for name in cases], fontsize=8)
    axis.set_ylabel('edge error, px RMS (painted truth)')
    axis.legend(fontsize=8)
    save(figure, 'edge_error')


def width_signal():
    """*Width statistics, Which width signal*: typical largest change inside one good line, mm."""
    sources = ('Band edges,\nsteepest slope', 'Band edges,\nprofile fit', 'Band edges,\ntrees',
               'SAM 2.1\noutline', 'Segmentation\nmask')
    values = {'dark': (1.34, 1.07, 0.84, 0.27, 0.16), 'pink': (1.69, 1.61, 0.96, 1.31, 0.56),
              'white': (0.74, 0.64, 0.54, 1.55, 0.56)}
    figure, axis = plt.subplots(figsize=(7.2, 2.9))
    width = 0.26
    for index, colour in enumerate(COLOURS):
        places = [number + (index - 1) * width for number in range(len(sources))]
        axis.bar(places, values[colour], width, label=colour, color=FABRIC[colour])
    axis.set_xticks(range(len(sources)), sources, fontsize=8)
    axis.set_ylabel('largest change in a good line, mm\n(over 40 px; lower is steadier)')
    axis.legend(title='fabric', fontsize=8)
    save(figure, 'width_signal')


def width_limit():
    """*Width statistics, Results*: smallest change marked significant, by how long it lasts."""
    lengths = (5, 11, 21, 42)
    limits = {'dark': (0.36, 0.27, 0.20, 0.14), 'pink': (0.58, 0.53, 0.44, 0.35), 'white': (0.76, 0.59, 0.51, 0.37)}
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    axis.axhspan(0.1, 0.2, color='#cfe8d4', label='target: 0.1-0.2 mm')
    for colour in COLOURS:
        axis.plot(lengths, limits[colour], marker='o', color=FABRIC[colour], label=colour, linewidth=2)
        axis.text(lengths[-1] + 0.8, limits[colour][-1], f'{limits[colour][-1]:.2f}', va='center', fontsize=8)
    axis.set_xscale('log')
    axis.set_xticks(lengths, [f'{length} mm\n({pixels} px)' for length, pixels in zip(lengths, (40, 80, 160, 320))])
    axis.minorticks_off()
    axis.set_xlim(4.3, 52)
    axis.set_ylim(0, 0.85)
    axis.set_xlabel('how long the change lasts along the line')
    axis.set_ylabel('smallest width change found, mm')
    axis.legend(fontsize=8, ncol=4, loc='upper right')
    save(figure, 'width_limit')


def false_alarms():
    """*Ground facts*: good lines falsely flagged of 257, by threshold."""
    thresholds = (4, 5, 6, 7, 10)
    both = (30, 8, 3, 1, 0)
    band = (223, 190, 156, 132, 102)
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    places = range(len(thresholds))
    axis.bar([place - 0.2 for place in places], band, 0.4, label='band edges alone', color='#c8553d')
    axis.bar([place + 0.2 for place in places], both, 0.4, label='SAM and band edges must agree', color='#2f6f4f')
    for place, (first, second) in enumerate(zip(band, both)):
        axis.text(place - 0.2, first + 4, str(first), ha='center', fontsize=8)
        axis.text(place + 0.2, second + 4, str(second), ha='center', fontsize=8)
    axis.set_xticks(list(places), [f'{value}{" (default)" if value == 7 else ""}' for value in thresholds])
    axis.set_xlabel('threshold, in spreads of the line\'s own noise')
    axis.set_ylabel('good lines flagged, of 257')
    axis.legend(fontsize=8)
    save(figure, 'false_alarms')


def widths_by_colour():
    """*The main branch's local model, Scored*: median width, px."""
    readings = {'Version 5 (this branch)': (31.0, 28.7, 31.3), 'Glue model on main': (31.0, 30.9, 32.6)}
    figure, axis = plt.subplots(figsize=(5.2, 2.6))
    width = 0.36
    for index, (name, values) in enumerate(readings.items()):
        places = [number + (index - 0.5) * width for number in range(3)]
        axis.bar(places, values, width, label=name, color=('#2f6f4f', '#7b5ea7')[index])
        for place, value in zip(places, values):
            axis.text(place, value + 0.15, f'{value:.1f}', ha='center', fontsize=8)
    axis.set_xticks(range(3), COLOURS)
    axis.set_ylim(26, 34.5)
    axis.set_ylabel('median line width, px\n(the same 4 mm strip)')
    axis.legend(fontsize=8, loc='upper left')
    save(figure, 'widths_by_colour')


def signal():
    """*Accuracy, What a real glue line's cross-section is*: band depth and texture share."""
    figure, (first, second) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    first.bar(COLOURS, (46.5, 7, 7), color=[FABRIC[colour] for colour in COLOURS])
    first.set_ylabel('band darker than fabric,\ngrey levels of 255')
    for place, value in enumerate(('43-50', '7', '7')):
        first.text(place, (46.5, 7, 7)[place] + 1.5, value, ha='center', fontsize=8)
    second.bar(COLOURS, (0.21, 0.24, 0.39), color=[FABRIC[colour] for colour in COLOURS])
    second.axhline(1.0, color='#888', linewidth=0.8, linestyle='--')
    second.text(2.45, 1.0, 'fabric', ha='right', va='bottom', fontsize=8, color='#666')
    second.set_ylim(0, 1.1)
    second.set_ylabel('weave texture inside the band,\nshare of the fabric\'s')
    save(figure, 'signal')


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    for draw in (time_per_stage, time_two_pipelines, edge_error, width_signal, width_limit, false_alarms, widths_by_colour, signal):
        draw()
