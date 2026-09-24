import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_reference_preparation import frame_rectangular_reference, prepare_references, validate_face


class ReferencePreparationTests(unittest.TestCase):
    def test_rectangular_crop_is_square_contains_face_and_does_not_squash(self):
        # Coordinate ramps expose unequal scaling of the horizontal/vertical axes.
        pixels = np.zeros((500, 300, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(300)[None, :] % 256
        pixels[:, :, 1] = np.arange(500)[:, None] % 256
        image = Image.fromarray(pixels)
        box = [110, 190, 190, 290]
        framed, info = frame_rectangular_reference(image, box)
        left, top, right, bottom = info['crop_box_in_oriented_source']
        self.assertEqual(right - left, bottom - top)
        self.assertTrue(left <= box[0] < box[2] <= right)
        self.assertTrue(top <= box[1] < box[3] <= bottom)
        self.assertEqual(framed.size, (512, 512))
        self.assertEqual(info['padding_left_top_right_bottom'], [0, 0, 0, 0])
        self.assertAlmostEqual(info['uniform_scale'], 512 / (right - left))
        center = np.asarray(framed)[256, 256, :2]
        self.assertTrue(np.all(np.abs(center.astype(int) - [150, 240]) <= 1))

    def test_edge_face_framing_includes_chin_and_padding_is_recorded(self):
        image = Image.new('RGB', (120, 300), (50, 70, 90))
        box = [5, 10, 110, 240]
        framed, info = frame_rectangular_reference(image, box)
        left, top, right, bottom = info['crop_box_in_oriented_source']
        self.assertTrue(left <= box[0] and right >= box[2])
        self.assertTrue(top <= box[1] and bottom >= box[3])
        self.assertGreater(sum(info['padding_left_top_right_bottom']), 0)
        self.assertTrue(np.all(np.asarray(framed) == [50, 70, 90]))

    def test_square_photos_keep_original_paths_and_bytes_after_validation(self):
        face = SimpleNamespace(bbox=[16, 10, 70, 75], det_score=.99)
        class Detector:
            def __init__(self): self.calls = 0
            def get(self, pixels):
                self.calls += 1
                self.assertions(pixels)
                return [face]
            @staticmethod
            def assertions(pixels):
                assert pixels.flags.c_contiguous
                assert pixels.shape == (96, 96, 3)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = []
            for index in range(3):
                path = directory / f'{index}.png'
                Image.new('RGB', (96, 96), (index, 30, 90)).save(path)
                paths.append(path)
            original_bytes = [path.read_bytes() for path in paths]
            detector = Detector()
            prepared, entries = prepare_references(paths, directory / 'prepared', detector)
            self.assertEqual(prepared, paths)
            self.assertEqual([path.read_bytes() for path in prepared], original_bytes)
            self.assertEqual(detector.calls, 3)
            self.assertTrue(all(entry['geometry']['operation'] == 'unchanged_square' for entry in entries))
            self.assertTrue(all(entry['original_sha256'] == entry['prepared_sha256'] for entry in entries))

    def test_no_multiple_small_or_invalid_face_has_usable_error(self):
        face = SimpleNamespace(bbox=[5, 5, 60, 65], det_score=.9)
        for faces, expected in [([], 'Found 0'), ([face, face], 'Found 2'),
                                ([SimpleNamespace(bbox=[5, 5, 15, 25], det_score=.9)], 'larger'),
                                ([SimpleNamespace(bbox=[5, 5, 60, 65], det_score=.1)], 'larger'),
                                ([SimpleNamespace(bbox=[5, float('nan'), 60, 65], det_score=.9)], 'unusable')]:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                validate_face(faces, (100, 100), 2)

    def test_rectangular_preparation_records_original_and_prepared_geometry(self):
        class Detector:
            def get(self, pixels):
                return [SimpleNamespace(bbox=[40, 70, 130, 200], det_score=.95)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = []
            for index in range(3):
                path = directory / f'{index}.png'
                Image.new('RGB', (180, 300), (index * 50, 70, 90)).save(path)
                paths.append(path)
            original_bytes = [path.read_bytes() for path in paths]
            prepared, entries = prepare_references(paths, directory / 'prepared', Detector())
            self.assertEqual([path.read_bytes() for path in paths], original_bytes)
            for index, (path, entry) in enumerate(zip(prepared, entries)):
                self.assertNotEqual(path, paths[index])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (512, 512))
                self.assertEqual(entry['oriented_source_size'], [180, 300])
                self.assertEqual(entry['geometry']['operation'], 'face_centered_square_crop')
                self.assertNotEqual(entry['original_sha256'], entry['prepared_sha256'])

    def test_duplicate_reference_files_are_rejected_before_detection(self):
        class Detector:
            def get(self, pixels):
                raise AssertionError('Duplicate rejection must happen before face detection.')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / 'same.png'
            Image.new('RGB', (96, 96)).save(path)
            with self.assertRaisesRegex(ValueError, 'different reference photographs'):
                prepare_references([path, path, path], directory / 'prepared', Detector())


if __name__ == '__main__':
    unittest.main()
