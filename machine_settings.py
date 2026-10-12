"""Settings of this PC, kept in Documents so an application update cannot reset them.

``config.json`` beside the code holds the defaults and is replaced by every update.
What an operator changes in the settings forms is stored as the *differences* from
those defaults in ``Documents/data files/Files/machine_settings.json`` and laid over
the defaults at start. Storing differences, not a full copy, means a default that a
later version changes still reaches this PC unless it was changed here.

Only the standard library and app_storage are imported: this runs before the
startup window, ahead of the camera and model modules.
"""
import copy
import json
import os
from pathlib import Path
import tempfile

DEFAULTS_PATH = Path(__file__).resolve().parent / 'config.json'
MACHINE_FILE = 'Files/machine_settings.json'


def machine_path():
    from app_storage import data_path
    return Path(data_path(MACHINE_FILE))


def _paired(first, second):
    """Two lists that can be compared item by item: same length, all dictionaries."""
    return (isinstance(first, list) and isinstance(second, list) and len(first) == len(second)
            and all(isinstance(item, dict) for item in first + second))


def difference(defaults, config):
    """What ``config`` changes in ``defaults``. Keys unknown to the defaults are dropped."""
    changed = {}
    for key, value in config.items():
        if key not in defaults:
            continue
        default = defaults[key]
        if isinstance(default, dict) and isinstance(value, dict):
            inner = difference(default, value)
            if inner:
                changed[key] = inner
        elif _paired(default, value):
            inner = [difference(a, b) for a, b in zip(default, value)]
            if any(inner):
                changed[key] = inner
        elif value != default or type(value) is not type(default):
            changed[key] = copy.deepcopy(value)
    return changed


def merged(defaults, changes):
    """The defaults with ``changes`` laid over them. Neither argument is modified."""
    result = copy.deepcopy(defaults)
    if not isinstance(changes, dict):
        return result
    for key, value in changes.items():
        if key not in result:
            continue                    # a setting a later version removed
        default = result[key]
        if isinstance(default, dict) and isinstance(value, dict):
            result[key] = merged(default, value)
        elif _paired(default, value):
            result[key] = [merged(a, b) for a, b in zip(default, value)]
        elif isinstance(default, dict) or isinstance(value, dict):
            continue                    # the shape changed: keep the default
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_defaults(defaults_path=DEFAULTS_PATH):
    return json.loads(Path(defaults_path).read_text(encoding='utf-8-sig'))


def load_changes(path=None):
    """This PC's saved differences; {} when there are none or the file is unreadable."""
    path = machine_path() if path is None else Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config(defaults_path=DEFAULTS_PATH, path=None, validate=None):
    """Defaults with this PC's changes. If the result is invalid, the defaults alone.

    ``validate`` raises ValueError for a bad configuration. A saved change can stop
    being valid when a later version changes what a setting means; the application
    must still start, so the defaults are used and the changes are left on disk.
    """
    defaults = load_defaults(defaults_path)
    config = merged(defaults, load_changes(path))
    if validate is not None and config != defaults:
        try:
            validate(config)
        except ValueError:
            return defaults
    return config


def save_config(config, defaults_path=DEFAULTS_PATH, path=None):
    """Store what ``config`` changes in the defaults. Returns the file written."""
    path = machine_path() if path is None else Path(path)
    changes = difference(load_defaults(defaults_path), config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as stream:
            temporary = stream.name
            json.dump(changes, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path
