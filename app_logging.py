"""Optional application log (``logging`` in config.json; off by default).

Off: nothing is written anywhere, as before. On: what the application does is
written to ``Documents/data files/logs/flod_mas.log`` (kept to a fixed size) and
to the console: start-up, where frames come from, the hand trigger, the fabric
check, every measured line and every error, including ones nothing else catches.

Modules log through ``logging.getLogger('flod.<area>')`` whether this is on or
not; without ``setup`` those calls go nowhere.
"""
import logging
from logging.handlers import RotatingFileHandler
import sys
import threading

ROOT = 'flod'
LOG_FILE = 'logs/flod_mas.log'
LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')
DEFAULTS = {'enabled': False, 'level': 'INFO', 'max_bytes': 5_000_000, 'backups': 5}

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def settings(config):
    """The ``logging`` section with defaults filled in. Raises ValueError if invalid."""
    section = config.get('logging', {})
    if not isinstance(section, dict):
        raise ValueError('logging must be a group of settings')
    merged = dict(DEFAULTS, **section)
    if not isinstance(merged['enabled'], bool):
        raise ValueError('logging.enabled must be true or false')
    if merged['level'] not in LEVELS:
        raise ValueError('logging.level must be one of ' + ', '.join(LEVELS))
    for key, minimum in (('max_bytes', 10_000), ('backups', 0)):
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'logging.{key} must be a whole number of at least {minimum}')
    return merged


def setup(config, path=None):
    """Start the log if it is enabled. Returns the file written to, or None."""
    chosen = settings(config)
    if not chosen['enabled']:
        return None
    if path is None:
        from app_storage import data_path
        path = data_path(LOG_FILE)
    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    layout = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s [%(threadName)s] %(message)s',
                               '%Y-%m-%d %H:%M:%S')
    for handler in (RotatingFileHandler(path, maxBytes=chosen['max_bytes'], backupCount=chosen['backups'],
                                        encoding='utf-8'),
                    logging.StreamHandler(sys.stderr)):
        handler.setFormatter(layout)
        logger.addHandler(handler)
    logger.setLevel(chosen['level'])
    logger.propagate = False

    # Errors that nothing catches would otherwise only reach a console nobody reads.
    def uncaught(kind, error, trace):
        logger.critical('Uncaught error', exc_info=(kind, error, trace))
        sys.__excepthook__(kind, error, trace)

    def uncaught_in_thread(args):
        logger.critical('Uncaught error in thread %s', getattr(args.thread, 'name', '?'),
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = uncaught
    threading.excepthook = uncaught_in_thread
    logger.info('Log started: %s (level %s)', path, chosen['level'])
    return path


def watch_tk(root):
    """Send errors raised inside Tk callbacks to the log as well as to the console."""
    logger = logging.getLogger(ROOT + '.ui')
    original = root.report_callback_exception

    def report(kind, error, trace):
        logger.error('Error in a window callback', exc_info=(kind, error, trace))
        original(kind, error, trace)

    root.report_callback_exception = report
