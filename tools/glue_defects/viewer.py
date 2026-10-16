"""Window: each line's glue width along its length against the sticker marks, per glue model.

    python -m tools.glue_defects.viewer [--data DIR] [--source NAME] [--check]

Left: the glue model ("Glue mask from"), the lines with their verdicts, and the check's
four settings. Right: the photo with the line, the width along the line as a share of
the width nearby with the limit, and the glue at the clicked place in the photo as it
is (turned only, never straightened), with the mask's edge.

Red: flagged by the check. Yellow: marked by a sticker. Grey bar: the glue edges are
too dark to see there, so that stretch is not judged.

``--check`` prints every source's verdicts and closes, without a window.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import DATA, MM_PER_PX, STICKER_REACH_PX, load_photos, photo_of
from .check import LIMIT, LOCAL_MM, MIN_MM, TIP_MM, VIS_MIN, judge

NAMES = {'segments': "main's method: 40 graded segments", 'app': 'app glue model (SegFormer, straightened band)'}


def sources(data):
    """{label: folder} of every glue model measured in ``data``."""
    found = {}
    for folder in sorted(Path(data).iterdir()):
        if (folder / 'widths.json').is_file():
            label = NAMES.get(folder.name) or folder.name.replace('lab_', 'pipeline-lab ').replace('sam_', 'SAM ')
            found[label] = folder
    order = lambda item: (item[1].name != 'segments', item[1].name != 'app', item[0])
    return dict(sorted(found.items(), key=order))


def load_lines(data, folder):
    photos = load_photos(data)
    widths = json.loads((folder / 'widths.json').read_text())
    seen = json.loads((Path(data) / 'visibility.json').read_text())
    lines = []
    for name, r in widths.items():
        info = photos[photo_of(name)]
        arc, w, xy = np.array(r['arc'], float), np.array(r['w'], float), np.array(r['xy'], float)
        arc = arc - arc[0]
        marks = []
        for sx, sy in info.get('stickers', []):
            d = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)
            if d.min() < STICKER_REACH_PX:
                i = int(d.argmin())
                marks.append((max(0., arc[i] - 15), min(arc[-1], arc[i] + 15)))
        edge = bool(xy[:, 0].min() < 20 or xy[:, 1].min() < 20)
        truth = ('cut by photo edge' if edge else 'sticker' if marks else
                 'defect-free' if info.get('good') else 'no sticker')
        vis = np.array(r['vis'] if 'vis' in r else seen[name], float)
        lines.append(dict(name=name, label=f"{info.get('label', photo_of(name))} line {name.rsplit('_L', 1)[1]}",
                          arc=arc, w=w, xy=xy, vis=vis, marks=marks, truth=truth, edge=edge,
                          origin=r.get('origin', [0, 0]), file=Path(data) / info['file']))
    return lines


def verdicts(lines, settings):
    """Per line its failing runs; and the counts the window's summary shows."""
    counts = dict(caught=0, sticker=0, failed=0, passed=0)
    out = []
    for line in lines:
        runs = judge(line['arc'], line['w'], line['vis'], cut_by_edge=line['edge'], **settings)[3]
        hit = any(a <= q and p <= b for a, b in line['marks'] for p, q, _ in runs)
        if line['truth'] == 'sticker':
            counts['sticker'] += 1
            counts['caught'] += hit
        elif line['truth'] != 'cut by photo edge':
            counts['failed' if runs else 'passed'] += 1
        out.append(runs)
    return out, counts


TAG = {'sticker': 'S', 'no sticker': '-', 'defect-free': 'G', 'cut by photo edge': 'E'}


def check(data):
    settings = dict(limit=LIMIT, min_mm=MIN_MM, local_mm=LOCAL_MM, vis_min=VIS_MIN)
    for label, folder in sources(data).items():
        lines = load_lines(data, folder)
        runs, c = verdicts(lines, settings)
        print(f"== {label}: sticker spots caught {c['caught']}/{c['sticker']}, unmarked lines failed {c['failed']}/"
              f"{c['failed'] + c['passed']}")
        for line, r in zip(lines, runs):
            print(f"   {'FAIL' if r else 'ok  '} [{TAG[line['truth']]}] {line['label']:40s} "
                  + ', '.join(f'{p:.0f}-{q:.0f} mm ({m:.2f})' for p, q, m in r))


def window(data, source=None):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    found = sources(data)

    class Viewer(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title('Glue width defects')
            self.geometry('1500x900')
            self.source = tk.StringVar(value=source if source in found else next(iter(found)))
            self.lines = load_lines(data, found[self.source.get()])
            self.photos = {}
            self.limit = tk.DoubleVar(value=LIMIT)
            self.min_mm = tk.DoubleVar(value=MIN_MM)
            self.local_mm = tk.DoubleVar(value=LOCAL_MM)
            self.vis_min = tk.DoubleVar(value=VIS_MIN)
            self.at = None
            left = ttk.Frame(self, padding=6)
            left.pack(side='left', fill='y')
            ttk.Label(left, text='Glue mask from', font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
            ttk.OptionMenu(left, self.source, self.source.get(), *found, command=self.change_source).pack(fill='x', pady=(0, 6))
            ttk.Label(left, text='Lines', font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
            self.listbox = tk.Listbox(left, width=50, height=26, font=('TkFixedFont', 10), exportselection=False)
            self.listbox.pack(fill='y', expand=True)
            self.listbox.bind('<<ListboxSelect>>', lambda e: self.show())
            box = ttk.LabelFrame(left, text='Check', padding=6)
            box.pack(fill='x', pady=6)
            for text, var, low, high, step in (('Fail below (share of the width nearby)', self.limit, .7, 1., .01),
                                               ('Over at least (mm)', self.min_mm, 2, 30, 1),
                                               ('Width nearby over (mm, 0 = whole line)', self.local_mm, 0, 200, 10),
                                               ('Judge only where edges are this sharp', self.vis_min, 0, 8, .5)):
                ttk.Label(box, text=text).pack(anchor='w')
                tk.Scale(box, variable=var, from_=low, to=high, resolution=step, orient='horizontal',
                         command=lambda v: self.refresh()).pack(fill='x')
            self.summary = ttk.Label(left, justify='left', font=('TkFixedFont', 10))
            self.summary.pack(anchor='w', pady=4)
            ttk.Label(left, justify='left', wraplength=360, foreground='#555',
                      text='Red: flagged.  Yellow: sticker.  Grey bar: too dark to see the edges, not judged.\n'
                           'Click the plot to see the glue there. mm uncalibrated.').pack(anchor='w')
            right = ttk.Frame(self, padding=6)
            right.pack(side='left', fill='both', expand=True)
            self.head = ttk.Label(right, font=('TkDefaultFont', 12, 'bold'))
            self.head.pack(anchor='w')
            top = ttk.Frame(right)
            top.pack(fill='x')
            self.board = tk.Label(top)
            self.board.pack(side='left')
            self.info = ttk.Label(top, justify='left', font=('TkFixedFont', 10), padding=10)
            self.info.pack(side='left', anchor='n')
            self.plot = tk.Canvas(right, height=230, bg='#1e1e1e', highlightthickness=0)
            self.plot.pack(fill='x', pady=4)
            self.plot.bind('<Button-1>', self.click)
            self.zoom_head = ttk.Label(right)
            self.zoom_head.pack(anchor='w')
            self.zoom = tk.Label(right, bg='#1e1e1e')
            self.zoom.pack(fill='x')
            self.refresh()
            self.listbox.selection_set(0)
            self.show()

        def settings(self):
            return dict(limit=self.limit.get(), min_mm=self.min_mm.get(), local_mm=self.local_mm.get(),
                        vis_min=self.vis_min.get())

        def change_source(self, _=None):
            self.lines = load_lines(data, found[self.source.get()])
            self.refresh()
            if not self.listbox.curselection():
                self.listbox.selection_set(0)
            self.show()

        def refresh(self):
            selected = self.listbox.curselection()
            self.listbox.delete(0, 'end')
            runs, c = verdicts(self.lines, self.settings())
            for line, r in zip(self.lines, runs):
                self.listbox.insert('end', f"{'FAIL' if r else 'ok  '} [{TAG[line['truth']]}] {line['label']}")
                self.listbox.itemconfig('end', foreground='#b00000' if r else '#006000')
            judged = np.mean([np.mean(judge(l['arc'], l['w'], l['vis'], cut_by_edge=l['edge'], **self.settings())[2])
                              for l in self.lines])
            self.summary.config(text=f"Sticker spots caught : {c['caught']} / {c['sticker']}\n"
                                     f'Line length judged   : {judged * 100:.0f} %\n'
                                     f"Unmarked lines failed: {c['failed']}\n"
                                     f"Unmarked lines passed: {c['passed']}\n"
                                     '[S] sticker  [-] no sticker  [G] known good\n[E] cut by photo edge (not judged)')
            if selected:
                self.listbox.selection_set(selected[0])
                self.show()

        def current(self):
            selected = self.listbox.curselection()
            return self.lines[selected[0] if selected else 0]

        def photo(self, path):
            if path not in self.photos:
                self.photos.clear()
                self.photos[path] = cv2.imread(str(path))
            return self.photos[path]

        def show(self):
            line = self.current()
            ref, ratio, judged, runs = judge(line['arc'], line['w'], line['vis'], cut_by_edge=line['edge'],
                                             **self.settings())
            self.runs, self.ratio = runs, ratio
            self.head.config(text=f"{line['label']}  ({line['truth']})   [{self.source.get()}]")
            img = self.photo(line['file'])
            small = cv2.resize(img, (576, 432), interpolation=cv2.INTER_AREA)
            f = 576 / img.shape[1]
            pts = np.rint(line['xy'] * f).astype(np.int32)
            cv2.polylines(small, [pts], False, (0, 255, 0), 2)
            cv2.circle(small, tuple(int(v) for v in pts[0]), 5, (255, 255, 255), -1)
            for p, q, _ in runs:
                sel = (line['arc'] >= p) & (line['arc'] <= q)
                cv2.polylines(small, [pts[sel]], False, (0, 0, 255), 5)
            for sx, sy in load_photos(data)[photo_of(line['name'])].get('stickers', []):
                cv2.circle(small, (int(sx * f), int(sy * f)), 9, (0, 220, 255), 2)
            self._board = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.board.config(image=self._board)
            text = [f"length        {line['arc'][-1]:6.0f} mm", f'median width  {float(np.median(ref)):6.2f} mm',
                    'white dot = 0 mm (start)', '']
            text += [f'FLAGGED {p:4.0f}-{q:4.0f} mm  min {m:.2f}x' for p, q, m in runs] or ['no narrow stretch']
            text += [''] + [f'sticker zone {a:4.0f}-{b:4.0f} mm' for a, b in line['marks']]
            self.info.config(text='\n'.join(text))
            if runs:
                self.at = (runs[0][0] + runs[0][1]) / 2
            elif line['marks']:
                self.at = sum(line['marks'][0]) / 2
            else:
                self.at = float(line['arc'][-1] / 2)
            self.draw_plot()
            self.draw_zoom()

        def draw_plot(self):
            line = self.current()
            c = self.plot
            c.delete('all')
            self.update_idletasks()
            W, H = max(c.winfo_width(), 900), 230
            L = line['arc'][-1]
            X = lambda a: 40 + a / L * (W - 60)
            Y = lambda v: 20 + (1.15 - min(max(v, .55), 1.15)) / .6 * (H - 50)
            for a, b in line['marks']:
                c.create_rectangle(X(a), 10, X(b), H - 30, fill='#5a5000', outline='')
            dark = line['vis'] < self.vis_min.get()
            start = None
            for i, d in enumerate(list(dark) + [False]):
                if d and start is None:
                    start = i
                elif not d and start is not None:
                    c.create_rectangle(X(line['arc'][start]), H - 44, X(line['arc'][i - 1]), H - 32, fill='#606060', outline='')
                    start = None
            for p, q, _ in self.runs:
                c.create_rectangle(X(p), 10, X(q), H - 30, fill='#5a0000', outline='')
            for v in (.6, .7, .8, .9, 1.0, 1.1):
                c.create_line(40, Y(v), W - 20, Y(v), fill='#333')
                c.create_text(20, Y(v), text=f'{v:.1f}', fill='#aaa', font=('TkDefaultFont', 8))
            c.create_line(40, Y(self.limit.get()), W - 20, Y(self.limit.get()), fill='#ff5050', dash=(4, 2))
            c.create_line(*np.column_stack([X(line['arc']), [Y(v) for v in self.ratio]]).ravel().tolist(),
                          fill='#40d0ff', width=2)
            for a in (TIP_MM, L - TIP_MM):
                c.create_line(X(a), 10, X(a), H - 30, fill='#777', dash=(2, 2))
            for mm in range(0, int(L) + 1, 25):
                c.create_text(X(mm), H - 18, text=str(mm), fill='#aaa', font=('TkDefaultFont', 8))
            c.create_text(W - 20, H - 6, text='mm along the line', anchor='e', fill='#aaa', font=('TkDefaultFont', 8))
            c.create_text(44, 12, anchor='w', fill='#ddd', font=('TkDefaultFont', 9),
                          text='width / width nearby, averaged over the "at least" length')
            c.create_line(X(self.at), 10, X(self.at), H - 30, fill='white')
            self._X = (40, W - 60, L)

        def click(self, event):
            x0, span, L = self._X
            self.at = float(np.clip((event.x - x0) / span * L, 0, L))
            self.draw_plot()
            self.draw_zoom()

        def draw_zoom(self):
            line = self.current()
            img = self.photo(line['file'])
            arc, xy = line['arc'], line['xy']
            a, b = max(0., self.at - 30), min(arc[-1], self.at + 30)
            at = lambda mm: np.array([np.interp(mm, arc, xy[:, 0]), np.interp(mm, arc, xy[:, 1])])
            P, A, B = at(self.at), at(a), at(b)
            # turned only, rigidly: bending the photo along the line would squeeze the band on curves
            angle = np.degrees(np.arctan2(B[1] - A[1], B[0] - A[0]))
            span = (b - a) / MM_PER_PX
            out_w, out_h = int(span + 80), int(max(160, .35 * span))
            M = cv2.getRotationMatrix2D((float(P[0]), float(P[1])), float(angle), 1.0)
            M[:, 2] += [out_w / 2 - P[0], out_h / 2 - P[1]]
            view = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
            mask_path = found[self.source.get()] / 'masks' / f"{line['name']}.png"
            mask = cv2.imread(str(mask_path), 0) if mask_path.is_file() else None
            if mask is not None:
                full = np.zeros(img.shape[:2], np.uint8)
                ox, oy = line['origin']
                h, w = min(mask.shape[0], full.shape[0] - oy), min(mask.shape[1], full.shape[1] - ox)
                full[oy:oy + h, ox:ox + w] = mask[:h, :w]
                contours, _ = cv2.findContours(cv2.warpAffine(full, M, (out_w, out_h), flags=cv2.INTER_NEAREST),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(view, contours, -1, (255, 0, 255), 1)
            to_view = lambda pts: np.c_[pts, np.ones(len(pts))] @ M.T

            def along(p, q, shift, colour):
                s = (arc >= max(p, a)) & (arc <= min(q, b))
                if s.sum() > 1:
                    cv2.polylines(view, [np.rint(to_view(xy[s]) + [0, shift]).astype(np.int32)], False, colour, 3)
            for p, q, _ in self.runs:
                along(p, q, -28, (0, 0, 255))
            for p, q in line['marks']:
                along(p, q, 28, (0, 220, 255))
            for mm in range(int(a // 5 * 5), int(b) + 1, 5):
                if a <= mm <= b:
                    x, y = to_view(at(mm)[None])[0]
                    cv2.line(view, (int(x), int(y) - 45), (int(x), int(y) - 36), (0, 255, 0), 1)
                    cv2.putText(view, str(mm), (int(x) + 2, int(y) - 48), 0, .4, (0, 255, 0), 1)
            scale = min(max(self.zoom.winfo_width(), 1100) / out_w, 300 / out_h)
            view = cv2.resize(view, (int(out_w * scale), int(out_h * scale)), interpolation=cv2.INTER_CUBIC)
            self._zoom = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            self.zoom.config(image=self._zoom)
            self.zoom_head.config(text=f'Photo as it is (turned only), {a:.0f}-{b:.0f} mm.  Magenta: glue mask edge, '
                                       'red above: flagged, yellow below: sticker zone')

    viewer = Viewer()
    viewer.lift()
    viewer.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--source', help='label of the glue model to open with')
    parser.add_argument('--check', action='store_true', help='print every source\'s verdicts, no window')
    args = parser.parse_args()
    if args.check:
        check(args.data)
    else:
        window(args.data, args.source)


if __name__ == '__main__':
    main()
