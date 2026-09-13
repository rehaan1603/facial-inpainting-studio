"""Boundary checks for the public-to-loopback deployment gateway, without GPU jobs."""
import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'webapp'))
from share_gateway import Gateway, Handler


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.server = Gateway(('127.0.0.1', 0), 8765, 'test-only-password-with-32-characters')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path='/', method='GET', authenticated=True, headers=None, body=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        all_headers = {'Authorization': self.server.authorization} if authenticated else {}
        all_headers.update(headers or {})
        connection.request(method, path, body, all_headers)
        response = connection.getresponse()
        status, payload = response.status, response.read()
        connection.close()
        return status, payload

    def test_all_paths_require_password(self):
        for path in ['/', '/app.js', '/api/session', '/runs/' + 'a'*32 + '/result.png']:
            self.assertEqual(self.request(path, authenticated=False)[0], 401)

    def test_no_dataset_or_old_run_or_filesystem_access(self):
        for path in ['/api/demo', '/api/demo?reference=1', '/configs/local.json',
                     '/runs/' + 'a'*32 + '/result.png', '/api/jobs/' + 'a'*32, '/../../README.md']:
            with patch.object(Handler, 'upstream') as upstream:
                self.assertEqual(self.request(path)[0], 404)
                upstream.assert_not_called()

    def test_public_session_never_returns_backend_token(self):
        with patch.object(Handler, 'upstream', return_value=(200, b'{"token":"private-backend","busy":false}', 'application/json')):
            status, payload = self.request('/api/session')
        self.assertEqual(status, 200)
        self.assertNotIn(b'private-backend', payload)
        self.assertFalse(json.loads(payload)['sample_available'])

    def test_csrf_and_invalid_upload_rejected_before_upstream(self):
        base = {'X-Local-Token': self.server.nonce, 'Content-Type': 'application/json'}
        with patch.object(Handler, 'upstream') as upstream:
            self.assertEqual(self.request('/api/inpaint', 'POST', headers={**base, 'Origin': 'https://evil.example'})[0], 403)
            self.assertEqual(self.request('/api/inpaint', 'POST', headers={'Content-Type': 'application/json'})[0], 403)
            self.assertEqual(self.request('/api/inpaint', 'POST', headers={**base, 'Content-Length': '16000001'})[0], 413)
            upstream.assert_not_called()

    def test_only_new_gateway_runs_become_accessible(self):
        job_id = 'b' * 32
        headers = {'X-Local-Token': self.server.nonce, 'Content-Type': 'application/json'}
        with patch.object(Handler, 'upstream', return_value=(202, json.dumps({'id': job_id}).encode(), 'application/json')):
            self.assertEqual(self.request('/api/inpaint', 'POST', headers=headers, body='{}')[0], 202)
        with patch.object(Handler, 'upstream', return_value=(200, b'png', 'image/png')):
            self.assertEqual(self.request('/runs/' + job_id + '/result.png')[0], 200)
            self.assertEqual(self.request('/runs/' + job_id + '/metadata.json')[0], 404)


if __name__ == '__main__':
    unittest.main()
