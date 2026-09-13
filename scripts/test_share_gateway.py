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
        status, page = self.request('/', authenticated=False)
        self.assertEqual(status, 200)
        self.assertIn(b'Demo password', page)
        for path in ['/app.js', '/api/session', '/runs/' + 'a'*32 + '/result.png']:
            self.assertEqual(self.request(path, authenticated=False)[0], 401)

    def test_form_login_sets_cookie_and_rejects_forgery(self):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request('POST', '/login', 'password=' + self.server.password,
                           {'Content-Type': 'application/x-www-form-urlencoded'})
        response = connection.getresponse()
        cookie = response.getheader('Set-Cookie')
        self.assertEqual(response.status, 303)
        response.read(); connection.close()
        self.assertIn('Secure; HttpOnly; SameSite=Strict', cookie)
        with patch.object(Handler, 'upstream', return_value=(200, b'{"busy":false}', 'application/json')):
            self.assertEqual(self.request('/api/session', authenticated=False, headers={'Cookie': cookie.split(';')[0]})[0], 200)
        self.assertEqual(self.request('/api/session', authenticated=False, headers={'Cookie': 'studio_session=9999999999.forged'})[0], 401)

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
