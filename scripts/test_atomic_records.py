"""Regression coverage for observed Windows progress-file locking failures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from atomic_records import write_json


class RecordTests(unittest.TestCase):
    def test_transient_lock_recovers_without_corrupting_record(self):
        original_replace = Path.replace
        attempts = []
        def locked_once(source, destination):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError('Simulated OneDrive reader lock')
            return original_replace(source, destination)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.json'
            path.write_text('{"step": 1}')
            with patch.object(Path, 'replace', locked_once), patch('atomic_records.time.sleep'):
                self.assertTrue(write_json(path, {'step': 2}))
            self.assertEqual(json.loads(path.read_text()), {'step': 2})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_optional_progress_does_not_abort_but_required_record_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.json'
            path.write_text('{"step": 1}')
            with patch.object(Path, 'replace', side_effect=PermissionError('Locked')), patch('atomic_records.time.sleep'):
                self.assertFalse(write_json(path, {'step': 2}, required=False))
                with self.assertRaises(PermissionError):
                    write_json(path, {'step': 2})
            self.assertEqual(json.loads(path.read_text()), {'step': 1})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
