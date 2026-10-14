"""One capture directory per detection; lossless sources and bounded retention."""
from datetime import datetime
from pathlib import Path
import atexit
import json
import queue
import re
import shutil
import threading
import uuid
import cv2

SET_NAME = re.compile(r'^\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$')


def _is_junction(path):
    """Path.is_junction() is Python 3.12+; junctions exist only on Windows."""
    check = getattr(path, 'is_junction', None)
    return bool(check()) if check else False


class BackgroundWriter:
    """Writes files on one thread, in the order they were handed over.

    Saving a 16-megapixel PNG takes over a second on the Jetson; with this the
    inspection carries on while the files are written. Order is kept, so a set's
    result.json (always its last write) appears only after all its images,
    which is what retention relies on. Pending writes are finished at exit.
    """

    def __init__(self):
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._run, name='capture-writer', daemon=True)
        self.thread.start()
        atexit.register(self.flush)

    def put(self, write, *args):
        self.jobs.put((write, args))

    def flush(self):
        """Block until everything handed over so far is on disk."""
        self.jobs.join()

    def _run(self):
        while True:
            write, args = self.jobs.get()
            try:
                write(*args)
            except Exception as error:
                # Nobody is waiting on the result any more; the log is the record.
                print(f'[storage] background save failed: {error}', flush=True)
            finally:
                self.jobs.task_done()


def _write_text(path, text):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def _write_image(path, image, options):
    if not cv2.imwrite(str(path), image, options):
        raise OSError(f'Could not save {path.name}')


class CaptureStore:
    def __init__(self, root, settings):
        self.root = Path(root).resolve()
        self.settings = settings
        self.lock = threading.RLock()
        # Off by default: saves finish before the caller continues, as always.
        self.writer = BackgroundWriter() if settings.get('background_save', False) else None
        self.root.mkdir(parents=True, exist_ok=True)
        # Recover this application's interrupted sets after a restart.
        for path in self._sets():
            if not (path / 'result.json').exists():
                self.finish(path, {'status': 'interrupted'})
        self.prune()

    def _sets(self):
        return sorted(p for p in self.root.iterdir()
                      if SET_NAME.fullmatch(p.name) and p.is_dir()
                      and not p.is_symlink() and not _is_junction(p)
                      and (p / '.inspection-set').is_file())

    def begin(self, original, undistorted, metadata):
        with self.lock:
            # Reserve space BEFORE creating the next set: total never exceeds limit.
            self.prune(reserve=1)
            name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
            path = self.root / name
            path.mkdir()
            (path / '.inspection-set').write_text('automatic-inspection-v1', encoding='utf-8')
            try:
                self._json(path / 'capture.json', metadata)
                self._image(path / 'original.png', original, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if undistorted is not None:
                    self._image(path / 'undistorted.png', undistorted, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            except Exception as error:
                self.finish(path, {'status': 'save_failed', 'error': str(error)})
                raise
            return path

    def save_preview(self, path, name, image):
        h, w = image.shape[:2]
        factor = min(1.0, self.settings['preview_max_edge'] / max(h, w))
        if factor < 1:
            image = cv2.resize(image, (max(1, round(w*factor)), max(1, round(h*factor))),
                               interpolation=cv2.INTER_AREA)
        self._image(path / f'{name}.jpg', image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.settings['jpeg_quality']])

    def finish(self, path, record):
        with self.lock:
            self._json(path / 'result.json', record)

    def save_mask(self, path, mask):
        self._image(path / 'mask.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def save_overlay(self, path, overlay):
        self._image(path / 'overlay.png', overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def update_metadata(self, path, metadata):
        with self.lock:
            self._json(path / 'capture.json', metadata)

    def _image(self, path, image, options):
        if self.writer is None:
            _write_image(path, image, options)
        else:
            # A private copy: the caller may reuse its array after this returns.
            self.writer.put(_write_image, path, image.copy(), options)

    def _json(self, path, record):
        # Serialised now, so later changes to ``record`` are not saved by mistake.
        text = json.dumps(record, indent=2, allow_nan=False)
        if self.writer is None:
            _write_text(path, text)
        else:
            self.writer.put(_write_text, path, text)

    def flush(self):
        """Wait for background saves (no-op when saving is synchronous)."""
        if self.writer is not None:
            self.writer.flush()

    def prune(self, reserve=0):
        with self.lock:
            paths = self._sets()
            remove_count = max(0, len(paths) + reserve - self.settings['max_sets'])
            for path in paths:
                if remove_count <= 0:
                    break
                if not (path / 'result.json').is_file():
                    continue  # never delete an inspection still being written
                resolved = path.resolve()
                if resolved.parent != self.root or path.is_symlink() or _is_junction(path):
                    raise OSError('Capture retention target escaped its storage directory')
                # Allow generated line folders, but reject links and unexpected subdirectories.
                if any(p.is_symlink() or _is_junction(p) or
                       (p.is_dir() and (p.parent != path or not re.fullmatch(r'line_\d+', p.name)))
                       for p in path.rglob('*')):
                    raise OSError(f'Unexpected content in capture set {path.name}')
                shutil.rmtree(resolved)
                remove_count -= 1
            if remove_count:
                raise OSError('Capture limit reached while another set is still being written')
