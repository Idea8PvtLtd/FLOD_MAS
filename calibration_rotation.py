"""A calibration is only valid for the camera rotation it was made with.

Frames are rotated right after capture (``cameras[n].rotation``), so the lens
calibration and the measurement plane both describe the *rotated* picture. If the
rotation setting is changed later, the saved lens centre, distortion and plane no
longer fit the picture, and nothing else would notice. New calibration files record
the rotation; loading refuses a file recorded with another one.

Files saved before this was added carry no record and are accepted as they are.
"""
KEY = 'camera_rotation'


class RotationMismatch(ValueError):
    pass


def normalise(rotation):
    return int(rotation or 0) % 360


def of_camera(camera_index):
    """The configured rotation of the camera with this device index, or None if unknown."""
    try:
        from app_config import camera_config
        return normalise(camera_config(camera_index).get('rotation', 0))
    except Exception:
        return None


def stamp(data, camera_index):
    """Record the camera's rotation in a calibration about to be saved."""
    rotation = of_camera(camera_index)
    if rotation is not None:
        data[KEY] = rotation
    return data


def check_file(path, rotation, label):
    """check() on a saved file. A file that cannot be read is left to its loader."""
    import json
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, ValueError, TypeError):
        return
    check(data, rotation, label)


def check(data, rotation, label):
    """Raise RotationMismatch when ``data`` was recorded with another rotation."""
    if not isinstance(data, dict) or data.get(KEY) is None:
        return
    recorded, current = normalise(data[KEY]), normalise(rotation)
    if recorded != current:
        raise RotationMismatch(
            f'The {label} was made with the camera picture turned {recorded} degrees, '
            f'but the camera is now set to {current} degrees. '
            'Calibrate the camera again, or set the rotation back.')
