"""A calibration made with one camera rotation is refused after the rotation changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import calibration_rotation
import camera_handler
import plane_scale

LENS = {'camera_matrix': [[2000, 0, 1000], [0, 2000, 800], [0, 0, 1]],
        'dist_coeffs': [0, 0, 0, 0, 0], 'image_size': [2000, 1600]}
SURFACE = {'rvec': [0.0, 0.0, 0.0], 'tvec': [0.0, 0.0, 0.5]}


class CheckTests(unittest.TestCase):
    def test_a_file_without_a_record_is_accepted(self):
        calibration_rotation.check(dict(LENS), 180, 'camera calibration')

    def test_the_recorded_rotation_must_match(self):
        calibration_rotation.check(dict(LENS, camera_rotation=180), 180, 'camera calibration')
        calibration_rotation.check(dict(LENS, camera_rotation=0), 360, 'camera calibration')
        with self.assertRaises(calibration_rotation.RotationMismatch) as raised:
            calibration_rotation.check(dict(LENS, camera_rotation=0), 180, 'camera calibration')
        self.assertIn('turned 0 degrees', str(raised.exception))
        self.assertIn('now set to 180', str(raised.exception))


class LoadingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.lens = Path(folder.name) / 'lens.json'
        self.surface = Path(folder.name) / 'surface.json'

    def write(self, lens_rotation=None, surface_rotation=None):
        lens, surface = dict(LENS), dict(SURFACE)
        if lens_rotation is not None:
            lens['camera_rotation'] = lens_rotation
        if surface_rotation is not None:
            surface['camera_rotation'] = surface_rotation
        self.lens.write_text(json.dumps(lens))
        self.surface.write_text(json.dumps(surface))

    def handler(self, rotation):
        handler = camera_handler.CameraHandler.__new__(camera_handler.CameraHandler)
        handler.calib_file_path = str(self.lens)
        handler.stream = type('Stream', (), {'rotation': rotation})()
        handler.reload_calibration()
        return handler

    def test_the_camera_refuses_a_lens_calibration_of_another_rotation(self):
        self.write(lens_rotation=0)
        self.assertTrue(self.handler(0).calibration_available)
        turned = self.handler(180)
        self.assertFalse(turned.calibration_available)
        self.assertIsNone(turned.undistorter)
        self.assertIn('Calibrate the camera again', turned.calibration_error)

    def test_the_camera_accepts_an_older_file_without_a_record(self):
        self.write()
        self.assertTrue(self.handler(180).calibration_available)

    def scale(self, rotation):
        config = {'cameras': [{'index': 0, 'rotation': rotation}]}
        with patch.object(plane_scale, '_paths_for', return_value=(self.lens, self.surface)), \
                patch('app_config.CONFIG', config):
            return plane_scale.load_frame_scale(1, (2000, 1600))

    def test_the_millimetre_scale_refuses_either_file_of_another_rotation(self):
        self.write(lens_rotation=180, surface_rotation=180)
        self.assertTrue(np.isfinite(self.scale(180).mm_per_pixel))
        for lens, surface in ((0, 180), (180, 0)):
            self.write(lens_rotation=lens, surface_rotation=surface)
            with self.assertRaises(plane_scale.PlaneScaleError) as raised:
                self.scale(180)
            self.assertIn('set the rotation back', str(raised.exception))

    def test_the_millimetre_scale_accepts_older_files(self):
        self.write()
        self.assertTrue(np.isfinite(self.scale(180).mm_per_pixel))


if __name__ == '__main__':
    unittest.main()
