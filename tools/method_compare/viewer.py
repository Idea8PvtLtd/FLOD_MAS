"""The comparison window: watch the recordings, compare edges up close, compare times.

    python -m tools.method_compare.viewer [RUN_FOLDER]

Opens the newest run under ~/flod_mas_data/method_compare when no folder is given.
Three tabs, sized for the 1366 x 768 screen the app uses:

  Watch      the video plays; at each still moment every method's lines appear
  Accuracy   one line close up, every method's two edges drawn over the same photo
  Speed      time per photo of each method against the 4 s target
"""
import sys
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import BY_ID, METHODS, REFERENCE, TARGET_SECONDS, latest_run
from .charts import mean_parts, nice_max, photo_parts, scale, summary, ticks, time_colour
from .results import load, match

try:
    from ui_theme import COLORS as C, FONT
except Exception:           # the viewer also runs without the app's theme
    C = {'bg': '#101722', 'surface': '#182230', 'surface_2': '#202D3E', 'card': '#1B2838',
         'border': '#34465D', 'text': '#F4F7FB', 'text_soft': '#CBD5E1', 'muted': '#8FA2BA',
         'accent': '#60A5FA', 'success': '#34D399', 'warning': '#FBBF24', 'danger': '#FB7185',
         'camera': '#030811'}
    FONT = 'DejaVu Sans'

WIDTH, HEIGHT = 1366, 730       # 768 minus the desktop's title bar
PANEL_W, PANEL_H = 226, 336
OPEN_ZOOM = 3.0                # Accuracy tab: screen pixels per photo pixel at first
SMALL_WIDTH = 920              # width of the cached copies of the still-moment photos
FPS = 10.0


def hex_colour(bgr):
    b, g, r = bgr
    return f'#{r:02x}{g:02x}{b:02x}'


def to_photo(image_bgr):
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))


def label(parent, text, size=10, colour=None, bold=False, **kwargs):
    return tk.Label(parent, text=text, bg=kwargs.pop('bg', C['bg']), fg=colour or C['text'],
                    font=(FONT, size, 'bold' if bold else 'normal'), **kwargs)


def flat_button(parent, text, command, width=None, accent=False):
    return tk.Button(parent, text=text, command=command, width=width, relief=tk.FLAT, bd=0,
                     bg=C['accent'] if accent else C['surface_2'], fg=C['bg'] if accent else C['text'],
                     activebackground=C['border'], activeforeground=C['text'], font=(FONT, 10, 'bold'),
                     padx=10, pady=4, cursor='hand2')


def apply_new_line_counts(methods, reference):
    """Version 5 photo times from the app's count of new lines on the board.

    Version 5 cannot tell a new line from an earlier one that a later cloth now
    partly covers (its middle moves), so its own count can be one too high. The
    app's BoardLines count is the one the app would act on: use it for the estimate.
    """
    if reference is None or reference.get('family') != 'app':
        return
    for data in methods.values():
        if data.get('family') != 'v5':
            continue
        for row, ref in zip(data['photos'], reference['photos']):
            row['photo_s'] = round(row['fixed_s'] + row['per_line_s'] * ref['new'], 3)
            row['new'] = ref['new']
        data['time_note'] = (f"photo = estimated: fixed stages + per-line share x lines new on this board "
                             f"(count from {reference['method']}); all_lines_s = measured, every line")


class Run:
    """Everything a run folder holds, loaded once."""

    def __init__(self, folder):
        self.folder = Path(folder)
        info = load(self.folder / 'moments.json')
        self.moments = info['moments']
        self.hand_note = info.get('hand_check', '')
        self.methods = {}
        for method in METHODS:
            path = self.folder / f"{method['id']}.json"
            if path.is_file():
                data = load(path)
                if len(data['photos']) == len(self.moments):
                    self.methods[method['id']] = data
        self.reference = self.methods.get(REFERENCE) or next(iter(self.methods.values()), None)
        apply_new_line_counts(self.methods, self.reference)
        # The playback timeline: every frame of every video, in order.
        self.frames, self.moment_at = [], {}
        for video in info['videos']:
            folder = self.folder / 'frames' / video['name']
            for path in sorted(folder.glob('*.jpg')):
                self.frames.append((video['name'], int(path.stem), path))
        position = {(name, frame): i for i, (name, frame, _) in enumerate(self.frames)}
        for index, moment in enumerate(self.moments):
            at = position.get((moment['video'], moment['frame']))
            if at is not None:
                self.moment_at[at] = index
        self._small, self._full = {}, {}

    def small(self, index):
        """(copy of the moment's photo SMALL_WIDTH wide, factor from full pixels)."""
        if index not in self._small:
            full = self.full(index)
            factor = SMALL_WIDTH / full.shape[1]
            self._small[index] = (cv2.resize(full, (SMALL_WIDTH, round(full.shape[0] * factor)),
                                             interpolation=cv2.INTER_AREA), factor)
        return self._small[index]

    def full(self, index):
        if index not in self._full:
            if len(self._full) > 2:
                self._full.pop(next(iter(self._full)))
            self._full[index] = cv2.imread(str(self.folder / 'photos' / self.moments[index]['photo']))
        return self._full[index]

    def lines(self, method_id, index):
        data = self.methods.get(method_id)
        return data['photos'][index]['lines'] if data else []


def draw_lines(image, lines, factor, colour, numbers=True):
    """A method's lines over a small photo: tinted outline, both edges, number."""
    out = image.copy()
    tint = image.copy()
    for line in lines:
        outline = np.rint(np.asarray(line['outline']) * factor).astype(np.int32)
        if len(outline) > 2:
            cv2.fillPoly(tint, [outline], colour)
    out = cv2.addWeighted(tint, .35, out, .65, 0)
    for line in lines:
        for side in ('left', 'right'):
            points = np.rint(np.asarray(line[side]) * factor).astype(np.int32)
            if len(points) > 1:
                cv2.polylines(out, [points], False, colour, 2, cv2.LINE_AA)
        if numbers and line['centre']:
            x, y = np.asarray(line['centre'][0]) * factor
            text = str(line.get('number') or '')
            cv2.putText(out, text, (int(x) - 8, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(out, text, (int(x) - 8, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class Viewer:
    def __init__(self, root, run):
        self.root, self.run = root, run
        root.title(f'Method comparison - {run.folder.name}')
        root.geometry(f'{WIDTH}x{HEIGHT}+0+0')
        root.configure(bg=C['bg'])
        self.position, self.playing, self.after_id = 0, False, None
        self.pause_at_moments = tk.BooleanVar(value=True)
        self.moment = 0 if run.moments else None
        self.zoom_moment, self.zoom_line = 0, 0
        self.shown = {m['id']: tk.BooleanVar(value=m['id'] in run.methods) for m in METHODS}
        self._images = {}
        self._build_header()
        self.pages = {name: tk.Frame(root, bg=C['bg']) for name in ('Watch', 'Accuracy', 'Speed')}
        self._build_watch(self.pages['Watch'])
        self._build_accuracy(self.pages['Accuracy'])
        self._build_speed(self.pages['Speed'])
        self.show_page('Watch')
        root.bind('<space>', lambda e: self.toggle_play())
        root.bind('<Right>', lambda e: self.next_moment(1))
        root.bind('<Left>', lambda e: self.next_moment(-1))
        self.seek(0)

    # ---------------------------------------------------------------- header
    def _build_header(self):
        bar = tk.Frame(self.root, bg=C['surface'], height=44)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        label(bar, 'Method comparison', 13, bold=True, bg=C['surface']).pack(side=tk.LEFT, padx=12)
        self.tab_buttons = {}
        for name in ('Watch', 'Accuracy', 'Speed'):
            button = flat_button(bar, name, lambda n=name: self.show_page(n), width=9)
            button.pack(side=tk.LEFT, padx=3, pady=7)
            self.tab_buttons[name] = button
        legend = tk.Frame(bar, bg=C['surface'])
        legend.pack(side=tk.RIGHT, padx=10)
        for method in METHODS:
            ran = method['id'] in self.run.methods
            tk.Label(legend, text='  ', bg=hex_colour(method['colour']) if ran else C['border']).pack(side=tk.LEFT, padx=(8, 3))
            label(legend, method['id'], 9, C['text'] if ran else C['muted'], bg=C['surface']).pack(side=tk.LEFT)

    def show_page(self, name):
        for page_name, page in self.pages.items():
            page.pack_forget()
            self.tab_buttons[page_name].configure(bg=C['accent'] if page_name == name else C['surface_2'],
                                                  fg=C['bg'] if page_name == name else C['text'])
        self.pages[name].pack(fill=tk.BOTH, expand=True)
        if name == 'Accuracy':
            self.root.after(30, self.render_zoom)
            self.draw_widths()
        elif name == 'Speed':
            self.draw_speed()

    # ----------------------------------------------------------------- watch
    def _build_watch(self, page):
        left = tk.Frame(page, bg=C['bg'])
        left.place(x=10, y=8, width=650, height=700)
        self.player = tk.Canvas(left, width=640, height=432, bg=C['camera'], highlightthickness=0)
        self.player.pack()
        controls = tk.Frame(left, bg=C['bg'])
        controls.pack(fill=tk.X, pady=6)
        flat_button(controls, '|< photo', lambda: self.next_moment(-1)).pack(side=tk.LEFT)
        self.play_button = flat_button(controls, 'Play', self.toggle_play, width=7, accent=True)
        self.play_button.pack(side=tk.LEFT, padx=6)
        flat_button(controls, 'photo >|', lambda: self.next_moment(1)).pack(side=tk.LEFT)
        tk.Checkbutton(controls, text='Pause at each photo', variable=self.pause_at_moments, bg=C['bg'],
                       fg=C['text'], selectcolor=C['surface_2'], activebackground=C['bg'],
                       activeforeground=C['text'], font=(FONT, 10)).pack(side=tk.LEFT, padx=12)
        self.timeline = tk.Canvas(left, width=640, height=46, bg=C['surface'], highlightthickness=0)
        self.timeline.pack()
        self.timeline.bind('<Button-1>', self._timeline_click)
        self.timeline.bind('<B1-Motion>', self._timeline_click)
        self.where = label(left, '', 10, C['text_soft'], anchor='w', justify=tk.LEFT)
        self.where.pack(fill=tk.X, pady=(6, 0))
        label(left, f"Still moments picked as the app's scene mode picks them. Hand check: {self.run.hand_note}.",
              8, C['muted'], anchor='w', justify=tk.LEFT, wraplength=630).pack(fill=tk.X, pady=(4, 0))
        grid = tk.Frame(page, bg=C['bg'])
        grid.place(x=668, y=6, width=690, height=PANEL_H * 2 + 8)
        self.panels = {}
        for i, method in enumerate(METHODS):
            canvas = tk.Canvas(grid, width=PANEL_W, height=PANEL_H, bg=C['card'], highlightthickness=1,
                               highlightbackground=C['border'])
            canvas.grid(row=i // 3, column=i % 3, padx=2, pady=2)
            self.panels[method['id']] = canvas
        self._draw_timeline()

    def _draw_timeline(self):
        t, frames = self.timeline, max(1, len(self.run.frames))
        t.delete('all')
        t.create_rectangle(8, 18, 632, 28, fill=C['surface_2'], outline='')
        previous = None
        for i, (video, _, _) in enumerate(self.run.frames):
            if video != previous and i:
                x = scale(i, 0, frames - 1, 8, 632)
                t.create_line(x, 10, x, 36, fill=C['border'])
            previous = video
        for at, index in self.run.moment_at.items():
            x = scale(at, 0, frames - 1, 8, 632)
            t.create_line(x, 14, x, 32, fill=C['accent'], width=3)
            t.create_text(x, 41, text=str(index + 1), fill=C['muted'], font=(FONT, 7))
        self.cursor = t.create_line(8, 6, 8, 36, fill=C['warning'], width=2)

    def _timeline_click(self, event):
        frames = max(1, len(self.run.frames))
        self.seek(int(round(scale(event.x, 8, 632, 0, frames - 1))))

    def seek(self, position):
        if not self.run.frames:
            return
        self.position = max(0, min(len(self.run.frames) - 1, position))
        video, frame, path = self.run.frames[self.position]
        image = cv2.imread(str(path))
        if image is not None:
            image = cv2.resize(image, (576, 432), interpolation=cv2.INTER_AREA)
            self._images['player'] = to_photo(image)
            self.player.delete('all')
            self.player.create_image(32, 0, image=self._images['player'], anchor='nw')
        frames = max(1, len(self.run.frames))
        x = scale(self.position, 0, frames - 1, 8, 632)
        self.timeline.coords(self.cursor, x, 6, x, 36)
        reached = [index for at, index in self.run.moment_at.items() if at <= self.position]
        latest = max(reached) if reached else None
        if latest != self.moment or not self._images.get('panels-drawn'):
            self.moment = latest
            self.draw_panels()
        text = f'{video}   frame {frame}   {frame / FPS:5.1f} s'
        if self.moment is not None:
            m = self.run.moments[self.moment]
            text += f"     showing photo {self.moment + 1} of {len(self.run.moments)} (frame {m['frame']}, board {m['board'].split('#')[-1]})"
        self.where.configure(text=text)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text='Pause' if self.playing else 'Play')
        if self.playing:
            self._tick()
        elif self.after_id:
            self.root.after_cancel(self.after_id)

    def _tick(self):
        if not self.playing:
            return
        if self.position >= len(self.run.frames) - 1:
            self.toggle_play()
            return
        self.seek(self.position + 1)
        if self.pause_at_moments.get() and self.position in self.run.moment_at:
            self.toggle_play()
            return
        self.after_id = self.root.after(int(1000 / FPS), self._tick)

    def next_moment(self, step):
        places = sorted(self.run.moment_at)
        if not places:
            return
        if step > 0:
            later = [at for at in places if at > self.position]
            self.seek(later[0] if later else places[-1])
        else:
            earlier = [at for at in places if at < self.position]
            self.seek(earlier[-1] if earlier else places[0])

    def draw_panels(self):
        self._images['panels-drawn'] = True
        for method in METHODS:
            canvas = self.panels[method['id']]
            canvas.delete('all')
            colour = hex_colour(method['colour'])
            canvas.create_rectangle(0, 0, PANEL_W, 4, fill=colour, outline='')
            canvas.create_text(8, 16, text=method['label'], anchor='w', fill=C['text'], font=(FONT, 9, 'bold'))
            data = self.run.methods.get(method['id'])
            if data is None:
                canvas.create_text(PANEL_W / 2, PANEL_H / 2, text='not run', fill=C['muted'], font=(FONT, 11))
                continue
            if self.moment is None:
                canvas.create_text(PANEL_W / 2, PANEL_H / 2, text='waiting for the first\nstill moment', fill=C['muted'],
                                   font=(FONT, 10), justify=tk.CENTER)
                continue
            row = data['photos'][self.moment]
            seconds = row['photo_s']
            state = time_colour(seconds, TARGET_SECONDS)
            estimate = ' est.' if data['family'] == 'v5' else ''
            canvas.create_text(8, 36, anchor='w', text=f'{seconds:.1f} s{estimate}', font=(FONT, 15, 'bold'),
                               fill=C['success'] if state == 'good' else C['danger'])
            note = (f"find {row['find_s']:.1f} + new {row['new_lines_s']:.1f} s" if data['family'] == 'app'
                    else f"all lines {row['all_lines_s']:.1f} s")
            canvas.create_text(218, 36, anchor='e', text=note, font=(FONT, 8), fill=C['muted'])
            small, factor = self.run.small(self.moment)
            picture = draw_lines(small, row['lines'], factor, method['colour'])
            picture = cv2.resize(picture, (222, round(picture.shape[0] * 222 / picture.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            key = f"panel-{method['id']}"
            self._images[key] = to_photo(picture)
            canvas.create_image(2, 50, image=self._images[key], anchor='nw')
            y = 50 + picture.shape[0] + 12
            for line in row['lines'][:6]:
                mm = f"  {line['width_mm']:.2f} mm" if line.get('width_mm') else ''
                new = '  NEW' if line.get('new') else ''
                canvas.create_text(8, y, anchor='w', font=(FONT, 9), fill=C['text_soft'],
                                   text=f"{line.get('number') or '-'}   {line['width_px']:.1f} px{mm}{new}")
                y += 16
            if row.get('errors'):
                canvas.create_text(8, y, anchor='w', font=(FONT, 8), fill=C['warning'],
                                   text=f"{len(row['errors'])} line(s) failed")

    # -------------------------------------------------------------- accuracy
    def _build_accuracy(self, page):
        top = tk.Frame(page, bg=C['bg'])
        top.place(x=10, y=6, width=1346, height=34)
        flat_button(top, '< photo', lambda: self.pick_zoom(self.zoom_moment - 1, 0)).pack(side=tk.LEFT)
        self.zoom_title = label(top, '', 10, C['text'])
        self.zoom_title.pack(side=tk.LEFT, padx=8)
        flat_button(top, 'photo >', lambda: self.pick_zoom(self.zoom_moment + 1, 0)).pack(side=tk.LEFT)
        self.line_buttons = tk.Frame(top, bg=C['bg'])
        self.line_buttons.pack(side=tk.LEFT, padx=14)
        toggles = tk.Frame(top, bg=C['bg'])
        toggles.pack(side=tk.RIGHT)
        for method in METHODS:
            tk.Checkbutton(toggles, text=method['id'], variable=self.shown[method['id']], command=self.render_zoom,
                           bg=C['bg'], fg=hex_colour(method['colour']), selectcolor=C['surface_2'],
                           activebackground=C['bg'], font=(FONT, 10, 'bold'),
                           state=tk.NORMAL if method['id'] in self.run.methods else tk.DISABLED).pack(side=tk.LEFT)
        self.zoom = tk.Canvas(page, width=850, height=600, bg=C['camera'], highlightthickness=0, cursor='fleur')
        self.zoom.place(x=10, y=46)
        self.zoom.bind('<ButtonPress-1>', self._drag_start)
        self.zoom.bind('<B1-Motion>', self._drag)
        for sequence, factor in (('<Button-4>', 1.25), ('<Button-5>', 0.8)):
            self.zoom.bind(sequence, lambda e, f=factor: self._zoom_by(f, e.x, e.y))
        self.zoom.bind('<MouseWheel>', lambda e: self._zoom_by(1.25 if e.delta > 0 else 0.8, e.x, e.y))
        label(page, 'Wheel: zoom   drag: move   Each method draws its two edges in its own colour; '
                    'B-fast dashed, as it reads exactly what B reads.',
              8, C['muted']).place(x=10, y=650)
        self.width_chart = tk.Canvas(page, width=480, height=630, bg=C['surface'], highlightthickness=0)
        self.width_chart.place(x=870, y=46)
        self.view = None
        self.pick_zoom(0, 0)

    def pick_zoom(self, moment, line):
        if not self.run.moments or self.run.reference is None:
            return
        self.zoom_moment = max(0, min(len(self.run.moments) - 1, moment))
        reference = self.run.lines(self.run.reference['method'], self.zoom_moment)
        self.zoom_line = max(0, min(len(reference) - 1, line)) if reference else 0
        for child in self.line_buttons.winfo_children():
            child.destroy()
        label(self.line_buttons, 'line', 10, C['muted']).pack(side=tk.LEFT, padx=(0, 4))
        for i, ref in enumerate(reference):
            flat_button(self.line_buttons, str(ref.get('number') or i + 1), lambda i=i: self.pick_zoom(self.zoom_moment, i),
                        accent=i == self.zoom_line).pack(side=tk.LEFT, padx=2)
        m = self.run.moments[self.zoom_moment]
        self.zoom_title.configure(text=f"photo {self.zoom_moment + 1} of {len(self.run.moments)}  ({m['video']}, frame {m['frame']})")
        self.view = None
        if reference:
            # Open on the middle of the line, close enough to see each method's edges apart.
            line = reference[self.zoom_line]
            centre = np.asarray(line['centre'], float)
            self.view = [centre[len(centre) // 2], OPEN_ZOOM]
        self.render_zoom()
        self.draw_widths()

    def _drag_start(self, event):
        self._drag_from = (event.x, event.y)

    def _drag(self, event):
        if self.view is None:
            return
        dx, dy = event.x - self._drag_from[0], event.y - self._drag_from[1]
        self._drag_from = (event.x, event.y)
        self.view[0] = self.view[0] - np.array([dx, dy]) / self.view[1]
        self.render_zoom()

    def _zoom_by(self, factor, x, y):
        if self.view is None:
            return
        centre, zoom = self.view
        at = centre + (np.array([x, y]) - np.array([425, 300])) / zoom
        zoom = float(np.clip(zoom * factor, 0.05, 12.0))
        self.view = [at - (np.array([x, y]) - np.array([425, 300])) / zoom, zoom]
        self.render_zoom()

    def render_zoom(self):
        canvas = self.zoom
        canvas.delete('all')
        if self.view is None or not self.run.moments:
            canvas.create_text(425, 300, text='No lines in this photo', fill=C['muted'], font=(FONT, 12))
            return
        photo = self.run.full(self.zoom_moment)
        (cx, cy), zoom = self.view
        half_w, half_h = 425 / zoom, 300 / zoom
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        x1, y1 = int(min(photo.shape[1], cx + half_w)), int(min(photo.shape[0], cy + half_h))
        if x1 > x0 and y1 > y0:
            crop = photo[y0:y1, x0:x1]
            size = (max(1, round((x1 - x0) * zoom)), max(1, round((y1 - y0) * zoom)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST if zoom > 2 else cv2.INTER_AREA)
            self._images['zoom'] = to_photo(crop)
            canvas.create_image((x0 - cx) * zoom + 425, (y0 - cy) * zoom + 300, image=self._images['zoom'], anchor='nw')
        reference = self.run.lines(self.run.reference['method'], self.zoom_moment)
        target = reference[self.zoom_line] if reference else None
        to_screen = lambda points: ((np.asarray(points) - (cx, cy)) * zoom + (425, 300)).ravel().tolist()
        for method in METHODS:
            if not self.shown[method['id']].get() or method['id'] not in self.run.methods:
                continue
            lines = self.run.lines(method['id'], self.zoom_moment)
            chosen = [lines[j] for i, j in match([target], lines)] if target is not None else []
            for line in chosen:
                for side in ('left', 'right'):
                    if len(line[side]) > 1:
                        canvas.create_line(*to_screen(line[side]), fill=hex_colour(method['colour']), width=2,
                                           dash=method.get('dash', ''))
        canvas.create_text(10, 12, anchor='w', fill=C['text'], font=(FONT, 9),
                           text=f'zoom {zoom:.2f} (screen pixels per photo pixel)')

    def draw_widths(self):
        c = self.width_chart
        c.delete('all')
        if self.run.reference is None or not self.run.moments:
            return
        c.create_text(14, 18, anchor='w', fill=C['text'], font=(FONT, 11, 'bold'), text='Width of each line in this photo')
        c.create_text(14, 38, anchor='w', fill=C['muted'], font=(FONT, 8),
                      text=f'px, mean along the line; grey bar = {REFERENCE} (the app today)')
        reference = self.run.lines(self.run.reference['method'], self.zoom_moment)
        values = [line['width_px'] for m in self.run.methods for line in self.run.lines(m, self.zoom_moment)]
        if not values:
            return
        low, high = np.floor(min(values) - 1), np.ceil(max(values) + 1)
        left, right = 60, 460
        for tick in np.linspace(low, high, 5):
            x = scale(tick, low, high, left, right)
            c.create_line(x, 54, x, 60 + 34 * len(reference), fill=C['border'])
            c.create_text(x, 64 + 34 * len(reference), text=f'{tick:.0f}', fill=C['muted'], font=(FONT, 8))
        for row, ref in enumerate(reference):
            y = 72 + row * 34
            c.create_text(30, y, text=str(ref.get('number') or row + 1), fill=C['text'], font=(FONT, 11, 'bold'))
            x = scale(ref['width_px'], low, high, left, right)
            c.create_line(x, y - 12, x, y + 12, fill=C['muted'], width=3)
            for k, method in enumerate(METHODS):
                lines = self.run.lines(method['id'], self.zoom_moment)
                for i, j in match([ref], lines):
                    x = scale(lines[j]['width_px'], low, high, left, right)
                    dy = (k - 2.5) * 3
                    c.create_oval(x - 5, y + dy - 5, x + 5, y + dy + 5, fill=hex_colour(method['colour']), outline='')
        y = 110 + 34 * len(reference)
        c.create_text(14, y, anchor='w', fill=C['text'], font=(FONT, 11, 'bold'), text='All photos')
        headers = ('method', 'lines', f'vs {REFERENCE}, px', 'same line, SD px')
        xs = (14, 120, 190, 330)
        for x, h in zip(xs, headers):
            c.create_text(x, y + 22, anchor='w', fill=C['muted'], font=(FONT, 8), text=h)
        for k, method in enumerate(METHODS):
            data = self.run.methods.get(method['id'])
            yy = y + 42 + k * 20
            c.create_text(xs[0], yy, anchor='w', fill=hex_colour(method['colour']), font=(FONT, 9, 'bold'), text=method['id'])
            if data is None:
                c.create_text(xs[1], yy, anchor='w', fill=C['muted'], font=(FONT, 9), text='not run')
                continue
            s = summary(data, self.run.moments, self.run.reference)
            diff = ('reference' if data is self.run.reference else
                    f"{s['difference_mean_px']:+.2f} ({s['difference_low_px']:+.1f} to {s['difference_high_px']:+.1f})"
                    if s['difference_mean_px'] is not None else '-')
            spread = f"{s['spread_median_px']:.2f} / {s['spread_max_px']:.2f}" if s['spread_median_px'] is not None else '-'
            for x, text in zip(xs[1:], (str(s['lines']), diff, spread)):
                c.create_text(x, yy, anchor='w', fill=C['text_soft'], font=(FONT, 9), text=text)
        note = ('The recordings have no ground truth. On painted lines with known edges (development PC), '
                'the trained-tree edges used by B, B-fast, C, D and E err by 0.63 px RMS; the plain '
                'steepest slope by about 1 px. SD = how much the same line reads from photo to photo.')
        c.create_text(14, y + 50 + 20 * len(METHODS), anchor='nw', width=450, fill=C['muted'], font=(FONT, 8), text=note)

    # ----------------------------------------------------------------- speed
    def _build_speed(self, page):
        self.speed = tk.Canvas(page, width=1346, height=672, bg=C['surface'], highlightthickness=0)
        self.speed.place(x=10, y=8)

    def draw_speed(self):
        c = self.speed
        c.delete('all')
        ran = [m for m in METHODS if m['id'] in self.run.methods]
        if not ran:
            c.create_text(673, 300, text='No method results in this run', fill=C['muted'], font=(FONT, 12))
            return
        c.create_text(16, 20, anchor='w', fill=C['text'], font=(FONT, 13, 'bold'),
                      text='Time to read a board photo, mean of all still moments')
        c.create_text(16, 42, anchor='w', fill=C['muted'], font=(FONT, 9),
                      text='App methods: find the lines + measure the new lines (measured). '
                           'Version 5: fixed stages + new lines (estimated from its per-line stage times).')
        top = nice_max(max(max(sum(mean_parts(self.run.methods[m['id']])) for m in ran), TARGET_SECONDS) * 1.1)
        left, right, y0, band = 230, 900, 70, 40
        for tick in ticks(top):
            x = scale(tick, 0, top, left, right)
            c.create_line(x, y0 - 4, x, y0 + band * len(ran), fill=C['border'])
            c.create_text(x, y0 + band * len(ran) + 12, text=f'{tick:g} s', fill=C['muted'], font=(FONT, 8))
        xt = scale(TARGET_SECONDS, 0, top, left, right)
        c.create_line(xt, y0 - 8, xt, y0 + band * len(ran) + 2, fill=C['warning'], dash=(5, 3), width=2)
        c.create_text(xt, y0 - 14, text=f'{TARGET_SECONDS:g} s target', fill=C['warning'], font=(FONT, 9))
        heads = ('all lines', 'first photo', 'peak RAM', 'GPU')
        for k, head in enumerate(heads):
            c.create_text(960 + k * 95, y0 - 14, text=head, fill=C['muted'], font=(FONT, 9))
        for i, method in enumerate(ran):
            data = self.run.methods[method['id']]
            y = y0 + i * band
            first, second = mean_parts(data)
            colour = hex_colour(method['colour'])
            c.create_text(16, y + band / 2 - 4, anchor='w', fill=C['text'], font=(FONT, 10, 'bold'), text=method['label'])
            x1, x2 = scale(first, 0, top, left, right), scale(first + second, 0, top, left, right)
            c.create_rectangle(left, y + 6, x1, y + band - 12, fill=colour, outline='')
            c.create_rectangle(x1, y + 6, x2, y + band - 12, fill=colour, outline='', stipple='gray50')
            c.create_text(x2 + 8, y + band / 2 - 3, anchor='w', fill=C['text'], font=(FONT, 10, 'bold'),
                          text=f'{first + second:.2f} s')
            rows = data['photos']
            all_lines = np.mean([r.get('new_lines_s', 0) + r.get('other_lines_s', 0) + r.get('find_s', 0)
                                 if data['family'] == 'app' else r['all_lines_s'] for r in rows])
            cells = (f'{all_lines:.2f} s', f"{data.get('first_photo_s') or 0:.1f} s",
                     f"{data.get('peak_rss_mb', 0) / 1024:.1f} GB",
                     f"{data['gpu_mb']} MB" if 'gpu_mb' in data else ', '.join(p.replace('ExecutionProvider', '') for p in data.get('providers', [])))
            for k, text in enumerate(cells):
                c.create_text(960 + k * 95, y + band / 2 - 4, text=text, fill=C['text_soft'], font=(FONT, 9))
        c.create_text(left, y0 + band * len(ran) + 30, anchor='w', fill=C['muted'], font=(FONT, 8),
                      text='solid: find the lines (app) / fixed stages (version 5)     striped: the new lines')
        # Per photo: one line per method across the still moments.
        y_top, y_bottom, x_left, x_right = 400, 620, 70, 1300
        tmax = nice_max(max(max(r['photo_s'] for r in self.run.methods[m['id']]['photos']) for m in ran) * 1.1)
        c.create_text(16, y_top - 24, anchor='w', fill=C['text'], font=(FONT, 12, 'bold'), text='Each still moment')
        for tick in ticks(tmax):
            y = scale(tick, 0, tmax, y_bottom, y_top)
            c.create_line(x_left, y, x_right, y, fill=C['border'])
            c.create_text(x_left - 8, y, anchor='e', text=f'{tick:g} s', fill=C['muted'], font=(FONT, 8))
        yt = scale(TARGET_SECONDS, 0, tmax, y_bottom, y_top)
        c.create_line(x_left, yt, x_right, yt, fill=C['warning'], dash=(5, 3), width=2)
        count = len(self.run.moments)
        for k, moment in enumerate(self.run.moments):
            x = scale(k, 0, max(1, count - 1), x_left + 20, x_right - 20)
            c.create_text(x, y_bottom + 14, text=f"{k + 1}", fill=C['muted'], font=(FONT, 8))
        for method in ran:
            rows = self.run.methods[method['id']]['photos']
            points = []
            for k, row in enumerate(rows):
                x = scale(k, 0, max(1, count - 1), x_left + 20, x_right - 20)
                y = scale(row['photo_s'], 0, tmax, y_bottom, y_top)
                points += [x, y]
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=hex_colour(method['colour']), outline='')
            if len(points) > 2:
                c.create_line(*points, fill=hex_colour(method['colour']), width=2)
        c.create_text(x_left, y_bottom + 32, anchor='w', fill=C['muted'], font=(FONT, 8),
                      text='still moment (1-4: first recording, 5-8: second); the dashed line is the 4 s target')


def grab(root, path):
    """Save what the window shows (for reports and for checking the layout)."""
    from PIL import ImageGrab
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(path)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', nargs='?', help='run folder (default: the newest one)')
    parser.add_argument('--show', choices=('Watch', 'Accuracy', 'Speed'), default='Watch')
    parser.add_argument('--moment', type=int, help='open at this still moment (1 = the first)')
    parser.add_argument('--line', type=int, default=1, help='Accuracy tab: this line of the photo (1 = the first)')
    parser.add_argument('--screenshot', help='save the window to this PNG after opening, then close')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    folder = Path(args.run) if args.run else latest_run()
    if folder is None or not (Path(folder) / 'moments.json').is_file():
        raise SystemExit('No prepared run found: run python -m tools.method_compare.prepare first')
    root = tk.Tk()
    viewer = Viewer(root, Run(folder))
    if args.moment:
        index = args.moment - 1
        at = [position for position, moment in viewer.run.moment_at.items() if moment == index]
        if at:
            viewer.seek(at[0])
        viewer.pick_zoom(index, args.line - 1)
    viewer.show_page(args.show)
    if args.screenshot:
        def save_and_close():
            grab(root, args.screenshot)
            root.destroy()
        root.after(2500, save_and_close)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
