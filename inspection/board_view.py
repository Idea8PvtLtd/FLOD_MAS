"""Every glue line of a board on one screen: verdict, board picture and one card per line.

Shown in place of the camera view by the scene capture mode. ``LineResultView``
draws one line per tab, so only one line is ever in sight and nothing says how
the board did as a whole. Here the whole board is redrawn from each payload:
a large verdict, the photo with every line outlined and numbered, and a card
per line, all visible together on the 1366 x 768 screen.

The picture keeps the zoom and pan of ``CropResultView`` (one tab, the board),
and the live camera side panel behaves as it does in ``FabricResultView``, so
the dashboard can hold either view in ``result_view``.
"""
import math
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from crop_result_view import (
    CropResultView, MAX_ZOOM, ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR,
)
from ui_theme import COLORS as C, FONT, button as themed_button
from .board_overlay import (
    aggregate_verdict, annotated_line, board_overview, entry_status, fit_card, line_is_upright,
)
from .result_view import FabricResultView

VERDICT_TEXT = {'PASS': 'PASS', 'FAIL': 'FAIL', 'WARNING': 'WARNING', 'EMPTY': 'NO GLUE LINE YET'}
VERDICT_COLORS = {'PASS': C['success'], 'FAIL': C['danger'], 'WARNING': C['warning'],
                  'EMPTY': C['muted']}
DARK_TEXT = '#0B1220'       # on the bright verdict colours

# The cards share a column of this width beside the board picture. Two columns
# of cards hold six lines without scrolling on the target screen.
CARDS_WIDTH = 404
# Used until the cards' frame has a real size (before it is first drawn).
DEFAULT_CARDS_HEIGHT = 600
CARD_GAP = 4
# A line usually spans the whole board, so "fit the line" alone would barely
# zoom. Clicking a card zooms at least this far in, centred on the line.
LINE_ZOOM = 2.5


def card_grid(count):
    """(columns, rows) of the card grid for ``count`` lines.

    Up to three lines get a full-width card each, four to six sit in two
    columns, and more than six (not expected on this machine) in three columns
    of smaller cards rather than running off the screen. Never fewer than two
    rows, so a single card is not stretched down the whole screen.
    """
    columns = 1 if count <= 3 else 2 if count <= 6 else 3
    return columns, max(2, math.ceil(count / columns))


def width_text(measurement):
    """The mean width for a card: millimetres, or pixels when no scale is known."""
    if measurement is None:
        return '—'
    if measurement.get('mean_width_mm') is not None:
        return f"{measurement['mean_width_mm']:.2f} mm"
    if measurement.get('mean_width_px') is not None:
        return f"{measurement['mean_width_px']:.1f} px"
    return '—'


def end_zone_text(measurement):
    """Widths of the two end zones, or '' when the measurement has none."""
    parts = []
    for segment in measurement.get('segments') or ():
        if not segment.get('end_zone'):
            continue
        if segment.get('average_width_mm') is not None:
            parts.append(f"{segment['average_width_mm']:.2f} mm")
        elif segment.get('average_width_px') is not None:
            parts.append(f"{segment['average_width_px']:.1f} px")
        else:
            parts.append('not measured')
    return 'Ends: ' + ' / '.join(parts) if parts else ''


def detail_lines(entry):
    """What a card says under the width: why it failed, the end zones, the target."""
    measurement = entry.get('measurement')
    if measurement is None:
        return [entry.get('error') or 'This line could not be measured.']
    status = entry_status(entry)
    lines = []
    if status == 'FAIL':
        reasons = list(measurement.get('fail_reasons') or ())
        segments = [s for c in measurement.get('components', ()) if not c.get('ignored')
                    for s in c.get('segments', ())]
        outside = sum(1 for s in segments if s.get('within_tolerance') is False)
        # A plain width failure has no reason in words, so it is counted here.
        if outside and not reasons:
            reasons.append(f'{outside} of {len(segments)} segments outside tolerance')
        lines += reasons
    elif status == 'WARNING':
        lines.append('No millimetre scale: width in pixels, not graded'
                     if not measurement.get('metric') else 'Some segments could not be measured')
    ends = end_zone_text(measurement)
    if ends:
        lines.append(ends)
    if measurement.get('metric'):
        lines.append(f"Target {measurement['target_width_mm']:g} ± {measurement['tolerance_mm']:g} mm")
    return lines


class BoardResultView(CropResultView):
    """Verdict, numbered board picture and a card per line; redrawn from each payload."""

    # The live camera side panel is the one FabricResultView draws; its two
    # methods need only the attributes this class creates under the same names.
    update_live_preview = FabricResultView.update_live_preview
    clear_live_overlay = FabricResultView.clear_live_overlay

    def __init__(self, parent, show_live_preview=True):
        self.show_live_preview = show_live_preview
        self._live_photo = None
        self._live_item = None
        self._paused = False
        # Kept for the dashboard, which sets it on any result view. A board's
        # grade never changes after it is shown, so it is never called here.
        self.on_grade_changed = lambda status: None
        self.payload = None
        self.entries = []
        self.verdict = 'EMPTY'
        self.selected_line = None
        self.cards = []             # one dict of widgets per line, in line order
        self._card_pictures = []    # annotated crop per line, made once per payload
        self._overview_scale = 1.0
        self._cards_size = (0, 0)
        super().__init__(parent, lambda: None, title='BOARD RESULTS')

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        header = tk.Frame(self, bg=C['bg'])
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.verdict_label = tk.Label(header, text=VERDICT_TEXT['EMPTY'], bg=VERDICT_COLORS['EMPTY'],
                                      fg=DARK_TEXT, font=(FONT, 24, 'bold'), padx=22, pady=2)
        self.verdict_label.pack(side=tk.LEFT)
        self.info_label = tk.Label(header, text='', bg=C['bg'], fg=C['text'],
                                   font=(FONT, 14, 'bold'), anchor='w')
        self.info_label.pack(side=tk.LEFT, padx=(16, 0))
        themed_button(header, '+', lambda: self.zoom_by(ZOOM_IN_FACTOR), role='quiet',
                      font_size=12, width=2, padx=6, pady=6).pack(side=tk.RIGHT)
        themed_button(header, '−', lambda: self.zoom_by(ZOOM_OUT_FACTOR), role='quiet',
                      font_size=12, width=2, padx=6, pady=6).pack(side=tk.RIGHT, padx=(0, 4))
        self.whole_board_button = themed_button(header, 'WHOLE BOARD', self.show_whole_board,
                                                role='secondary', font_size=10, padx=12, pady=6)
        self.whole_board_button.pack(side=tk.RIGHT, padx=(0, 6))
        self._countdown_label = tk.Label(header, text='Click a card to zoom to its line',
                                         bg=C['bg'], fg=C['muted'], font=(FONT, 10))
        self._countdown_label.pack(side=tk.RIGHT, padx=(8, 12))

        body = tk.Frame(self, bg=C['bg'])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        if self.show_live_preview:
            # Live camera context only, never part of the saved results. Same
            # panel as FabricResultView, so the dashboard places its progress
            # widgets in it the same way.
            self.live_panel = tk.Frame(body, bg=C['surface'], width=240)
            self.live_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            self.live_panel.pack_propagate(False)
            tk.Label(self.live_panel, text='LIVE CAMERA', font=(FONT, 10, 'bold'),
                     bg=C['surface'], fg=C['accent']).pack(anchor='w', padx=12, pady=12)
            self.live_canvas = tk.Canvas(self.live_panel, height=180, bg=C['camera'],
                                         highlightthickness=0)
            self.live_canvas.pack(fill=tk.X, padx=10)
            tk.Label(self.live_panel, text='Hand monitoring stays active.\n'
                     'Add or remove fabric, then\nmove your hands away.',
                     bg=C['surface'], fg=C['muted'], justify='left',
                     font=(FONT, 10)).pack(anchor='w', padx=12, pady=14)
        self.cards_frame = tk.Frame(body, bg=C['bg'], width=CARDS_WIDTH)
        self.cards_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.cards_frame.pack_propagate(False)
        self.cards_frame.grid_propagate(False)
        self.cards_frame.bind('<Configure>', self._on_cards_configure)
        self.empty_label = tk.Label(self.cards_frame, text='No glue line has been\nmeasured on this board yet.',
                                    bg=C['bg'], fg=C['muted'], font=(FONT, 12), justify='center')
        self.canvas = tk.Canvas(body, bg=C['camera'], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        # X11 reports the wheel as buttons 4 and 5 instead of <MouseWheel>.
        self.canvas.bind('<Button-4>', lambda event: self._on_wheel_step(event, ZOOM_IN_FACTOR))
        self.canvas.bind('<Button-5>', lambda event: self._on_wheel_step(event, ZOOM_OUT_FACTOR))
        self.canvas.bind('<ButtonPress-1>', self._on_pan_start)
        self.canvas.bind('<B1-Motion>', self._on_pan_drag)
        self.canvas.bind('<Configure>', self._on_configure)

    def _on_wheel_step(self, event, factor):
        tab = self._active_tab()
        if tab is not None:
            self._zoom_at(tab, factor, (event.x, event.y))
        return 'break'

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def show_board(self, payload):
        """Redraw everything from ``payload``; called again as the board gains lines.

        Nothing is kept from the payload before, so showing the same payload
        twice gives the same screen. The picture goes back to the whole board:
        the new photo is what the operator needs to see first.
        """
        self.payload = payload
        self.entries = list(payload.get('lines') or ())
        self.verdict = payload.get('verdict') or aggregate_verdict(self.entries)
        self.selected_line = None
        source = payload.get('source_image')
        overview = payload.get('overview')
        if overview is None and source is not None:
            overview = board_overview(source, self.entries)
        # Line positions are in photo pixels; this takes them to the overview.
        self._overview_scale = (overview.shape[1] / float(source.shape[1])
                                if overview is not None and source is not None else 1.0)
        self.verdict_label.configure(text=VERDICT_TEXT.get(self.verdict, self.verdict),
                                     bg=VERDICT_COLORS.get(self.verdict, C['muted']))
        self.info_label.configure(text=self._info_text())
        self._card_pictures = [annotated_line(entry) for entry in self.entries]
        self._build_cards()
        self.show_crops([] if overview is None else [overview])

    def show_whole_board(self):
        """Back from a zoomed line to the whole board."""
        self.selected_line = None
        self._mark_selected()
        self.reset_view()

    def zoom_to_line(self, line_id):
        """Zoom and centre the board picture on one line (a click on its card)."""
        entry = next((e for e in self.entries if e.get('line_id') == line_id), None)
        tab = self._active_tab()
        if entry is None or tab is None or tab['image'] is None:
            return
        points = entry.get('outline')
        if points is None or len(points) == 0:
            points = entry.get('centre')
        if points is None or len(points) == 0:
            return
        points = np.asarray(points, dtype=float).reshape(-1, 2) * self._overview_scale
        low, high = points.min(axis=0), points.max(axis=0)
        self._fit_tab(tab)
        canvas_width, canvas_height = self._canvas_size()
        fitted = min(canvas_width / max(1.0, high[0] - low[0]),
                     canvas_height / max(1.0, high[1] - low[1])) * .9
        tab['zoom'] = min(MAX_ZOOM, max(LINE_ZOOM, fitted / max(1e-9, tab['fit'])))
        scale = self._scale(tab)
        middle = (low + high) / 2.0
        tab['offset'] = [float(canvas_width / 2.0 - middle[0] * scale),
                         float(canvas_height / 2.0 - middle[1] * scale)]
        self.selected_line = line_id
        self._mark_selected()
        self._render()

    def set_paused(self, paused):
        """The dashboard tells every result view; a board has nothing to unlock."""
        self._paused = bool(paused)

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------
    def _info_text(self):
        count = len(self.entries)
        parts = [f"{count} line{'' if count == 1 else 's'}"]
        new = [e for e in self.entries if e.get('new')]
        if new:
            parts.append(f'{len(new)} new')
        seconds = self.payload.get('seconds')
        if seconds is not None:
            parts.append(f'last photo {seconds:.1f} s')
        return '  ·  '.join(parts)

    # ------------------------------------------------------------------
    # cards
    # ------------------------------------------------------------------
    def _cards_area(self):
        """Size of the cards' frame, or the expected size before it is drawn."""
        width, height = self.cards_frame.winfo_width(), self.cards_frame.winfo_height()
        if width < 50 or height < 50:
            return CARDS_WIDTH, DEFAULT_CARDS_HEIGHT
        return width, height

    def _on_cards_configure(self, event):
        # The first real size arrives after the cards were built from the
        # expected one; rebuild so the pictures fit what is really there.
        if event.width < 50 or event.height < 50 or (event.width, event.height) == self._cards_size:
            return
        if self.entries:
            self._build_cards()

    def _build_cards(self):
        for child in self.cards_frame.winfo_children():
            if child is not self.empty_label:
                child.destroy()
        self.cards = []
        area_width, area_height = self._cards_area()
        self._cards_size = (area_width, area_height)
        # Rows and columns left over from a board with more lines must not keep space.
        columns_before, rows_before = self.cards_frame.grid_size()
        for column in range(columns_before):
            self.cards_frame.grid_columnconfigure(column, weight=0, uniform='')
        for row in range(rows_before):
            self.cards_frame.grid_rowconfigure(row, weight=0, uniform='')
        if not self.entries:
            self.empty_label.place(relx=.5, rely=.4, anchor='center')
            return
        self.empty_label.place_forget()
        columns, rows = card_grid(len(self.entries))
        card_width = area_width // columns - CARD_GAP
        card_height = area_height // rows - CARD_GAP
        for column in range(columns):
            self.cards_frame.grid_columnconfigure(column, weight=1, uniform='card')
        for row in range(rows):
            self.cards_frame.grid_rowconfigure(row, weight=1, uniform='card')
        for index, entry in enumerate(self.entries):
            card = self._make_card(entry, self._card_pictures[index], card_width, card_height)
            card['frame'].grid(row=index // columns, column=index % columns, sticky='nsew',
                               padx=(0 if index % columns == 0 else CARD_GAP, 0),
                               pady=(0 if index < columns else CARD_GAP, 0))
            self.cards.append(card)
        self._mark_selected()

    def _make_card(self, entry, picture, card_width, card_height):
        status = entry_status(entry)
        color = VERDICT_COLORS[status]
        line_id = entry.get('line_id')
        upright = line_is_upright(entry)
        # More than six lines: the cards are small, so the text takes less room.
        compact = card_width < 150
        frame = tk.Frame(self.cards_frame, bg=C['card'], highlightthickness=2,
                         highlightbackground=color, highlightcolor=color, cursor='hand2')
        frame.pack_propagate(False)
        inner_width, inner_height = card_width - 4, card_height - 4
        # An upright line gets a tall strip down the left of its card, a line
        # lying sideways a wide strip across the top.
        if upright:
            size = (max(24, min(64, inner_width // 4)), max(24, inner_height - 8))
            text_width = inner_width - size[0] - 18
        else:
            size = (max(24, inner_width - 8), max(24, min(56, inner_height // 4)))
            text_width = inner_width - 14
        image = fit_card(picture, size, upright)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        image_label = tk.Label(frame, image=photo, bg=C['camera'], bd=0)
        image_label.pack(side=tk.LEFT if upright else tk.TOP, padx=4, pady=4,
                         anchor='n' if upright else 'w')
        text = tk.Frame(frame, bg=C['card'])
        text.pack(side=tk.LEFT if upright else tk.TOP, fill=tk.BOTH, expand=True,
                  padx=(2, 6), pady=(4, 4))
        top = tk.Frame(text, bg=C['card'])
        top.pack(fill=tk.X)
        # The same number, on the same colour, as the badge on the board picture.
        number = tk.Label(top, text=str(line_id), bg=color, fg=DARK_TEXT, width=2,
                          font=(FONT, 12 if compact else 15, 'bold'))
        number.pack(side=tk.LEFT)
        title = tk.Label(top, text='' if compact else 'LINE', bg=C['card'], fg=C['muted'],
                         font=(FONT, 9, 'bold'))
        title.pack(side=tk.LEFT, padx=(6, 0))
        new_label = None
        if entry.get('new'):
            new_label = tk.Label(top, text='NEW', bg=C['accent'], fg=DARK_TEXT,
                                 font=(FONT, 8, 'bold'), padx=4)
            new_label.pack(side=tk.RIGHT)
        status_label = tk.Label(text, text=status, bg=C['card'], fg=color, anchor='w',
                                font=(FONT, 12 if compact else 15, 'bold'))
        status_label.pack(fill=tk.X, pady=(4, 0))
        width_label = tk.Label(text, text=width_text(entry.get('measurement')), bg=C['card'],
                               fg=C['text'], anchor='w', font=(FONT, 13 if compact else 19, 'bold'))
        width_label.pack(fill=tk.X)
        caption = tk.Label(text, text='mean width', bg=C['card'], fg=C['muted'], anchor='w',
                           font=(FONT, 8))
        caption.pack(fill=tk.X)
        detail_label = tk.Label(text, text='\n'.join(detail_lines(entry)), bg=C['card'],
                                fg=C['text_soft'], anchor='nw', justify='left',
                                font=(FONT, 8 if compact else 9), wraplength=max(60, text_width))
        detail_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        widgets = [frame, image_label, text, top, number, title, status_label, width_label,
                   caption, detail_label] + ([new_label] if new_label is not None else [])
        for widget in widgets:
            widget.bind('<Button-1>', lambda event, chosen=line_id: self.zoom_to_line(chosen))
        return {'frame': frame, 'line_id': line_id, 'status': status_label, 'width': width_label,
                'detail': detail_label, 'number': number, 'new': new_label, 'image': image_label,
                'photo': photo, 'color': color}

    def _mark_selected(self):
        """The card of the zoomed line gets the accent border; the rest their verdict's."""
        for card in self.cards:
            chosen = card['line_id'] == self.selected_line
            color = C['text'] if chosen else card['color']
            card['frame'].configure(highlightbackground=color, highlightcolor=color,
                                    bg=C['card_hover'] if chosen else C['card'])
