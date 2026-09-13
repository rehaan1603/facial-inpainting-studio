"""Password-protected, loopback proxy for a temporary research demonstration.

Only results created through this gateway are served. Dataset samples and older
local runs are never forwarded. Put an HTTPS tunnel in front; do not bind publicly.
"""
import argparse
import base64
import hmac
import hashlib
import json
import re
import secrets
import threading
import time
from http.cookies import SimpleCookie
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, upstream_port, password):
        super().__init__(address, Handler)
        self.upstream_port = upstream_port
        self.authorization = 'Basic ' + base64.b64encode(('studio:' + password).encode()).decode()
        self.nonce = secrets.token_urlsafe(32)
        self.password = password
        self.cookie_key = secrets.token_bytes(32)
        self.jobs = set()
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        pass  # Do not retain URLs, credentials or personal image data in access logs.

    def send(self, status, body, content_type='application/json', challenge=False, extra=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        for key, value in {
            'Content-Type': content_type, 'Content-Length': str(len(body)),
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'no-referrer', 'Cross-Origin-Resource-Policy': 'same-origin',
            'Content-Security-Policy': "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        }.items():
            self.send_header(key, value)
        if challenge:
            self.send_header('WWW-Authenticate', 'Basic realm="Private Inpainting Studio", charset="UTF-8"')
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def authenticated(self):
        supplied = self.headers.get('Authorization', '')
        if hmac.compare_digest(supplied.encode(), self.server.authorization.encode()):
            return True
        try:
            cookies = SimpleCookie(self.headers.get('Cookie', ''))
            expiry, signature = cookies['studio_session'].value.split('.')
            expected = hmac.new(self.server.cookie_key, expiry.encode(), hashlib.sha256).hexdigest()
            if int(expiry) > time.time() and hmac.compare_digest(signature, expected):
                return True
        except (KeyError, ValueError):
            pass
        if self.command == 'GET' and self.path in ['/', '/index.html', '/login']:
            self.login_page()
        else:
            self.send(401, {'error': 'Open the site and sign in with the demo password.'})
        return False

    def login_page(self, failed=False):
        message = 'Incorrect password. Please try again.' if failed else 'Enter the demo password shared by the owner.'
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Sign in · Inpainting Studio</title><link rel="stylesheet" href="/login.css"></head><body><main><h1>Inpainting Studio</h1><h2>Private research demo</h2><p>' + message + '</p>'
                '<form method="post" action="/login"><label for="password">Demo password</label> '
                '<input id="password" name="password" type="password" autocomplete="current-password" required autofocus maxlength="256"> '
                '<button type="submit">Open studio</button></form><p>Images are processed and saved on the host laptop. '
                'The laptop must remain awake and connected.</p></main></body></html>')
        self.send(200 if not failed else 401, page.encode(), 'text/html; charset=utf-8')

    def login(self):
        origin = self.headers.get('Origin')
        if origin and origin != f'https://{self.headers.get("Host")}':
            return self.send(403, {'error': 'Cross-origin sign-in is disabled.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1024 or self.headers.get('Transfer-Encoding'):
                return self.send(400, {'error': 'Invalid sign-in request.'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/x-www-form-urlencoded':
                return self.send(415, {'error': 'Use the sign-in form.'})
            supplied = parse_qs(self.rfile.read(size).decode()) .get('password', [''])[0]
            if not hmac.compare_digest(supplied.encode(), self.server.password.encode()):
                return self.login_page(failed=True)
            expiry = str(int(time.time()) + 8*60*60)
            signature = hmac.new(self.server.cookie_key, expiry.encode(), hashlib.sha256).hexdigest()
            return self.send(303, b'', extra={'Location': '/', 'Set-Cookie':
                f'studio_session={expiry}.{signature}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=28800'})
        except (ValueError, UnicodeError):
            return self.send(400, {'error': 'Invalid sign-in request.'})

    def upstream(self, method, path, body=None):
        con = HTTPConnection('127.0.0.1', self.server.upstream_port, timeout=30)
        try:
            headers = {}
            if method == 'POST':
                con.request('GET', '/api/session')
                response = con.getresponse()
                session = json.loads(response.read())
                if response.status != 200:
                    raise OSError('Upstream session unavailable')
                headers = {'Content-Type': 'application/json', 'X-Local-Token': session['token'],
                           'Origin': f'http://127.0.0.1:{self.server.upstream_port}'}
            con.request(method, path, body, headers)
            response = con.getresponse()
            return response.status, response.read(), response.getheader('Content-Type', 'application/json')
        finally:
            con.close()

    def do_GET(self):
        if self.path == '/login.css':
            return self.send(200, b'body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f4f3ef;color:#222;font:16px system-ui}main{max-width:420px;margin:24px;padding:40px;background:white;border:1px solid #deddd7;border-radius:18px}h1{font-size:28px}h2{font-size:16px;font-weight:500;color:#666}p{line-height:1.6;color:#666}label,input,button{display:block;box-sizing:border-box;width:100%}input{margin:10px 0 18px;padding:14px;border:1px solid #aaa;border-radius:8px;font:inherit}button{background:#222;color:white;border:0;border-radius:8px;padding:14px;font:inherit;cursor:pointer}', 'text/css')
        if not self.authenticated():
            return
        path = self.path
        allowed = path in ['/', '/index.html', '/app.js', '/style.css', '/favicon.svg', '/api/session']
        match = re.fullmatch(r'/(?:api/jobs/|runs/)([a-f0-9]{32})(?:/(?:input|result|effective_mask)\.png)?', path)
        if match:
            with self.server.lock:
                allowed = match[1] in self.server.jobs
        if not allowed:
            return self.send(404, {'error': 'Upload your own photo. Local datasets and earlier runs are not shared.'})
        try:
            status, payload, kind = self.upstream('GET', path)
            if status == 200 and path == '/api/session':
                data = json.loads(payload)
                payload = json.dumps({'token': self.server.nonce, 'busy': data['busy'],
                                      'shared': True, 'sample_available': False}).encode()
            if status == 200 and path in ['/', '/index.html']:
                page = payload.decode('utf-8')
                page = page.replace('Runs on your laptop', "Runs on the host's GPU")
                page = page.replace('Your image stays on this laptop.', 'Uploads travel through Cloudflare to the host laptop and are saved there for this research demo.')
                page = page.replace('No account. No cloud upload.', 'Private demo · Upload only photos you have permission to use.')
                payload = page.encode('utf-8')
            return self.send(status, payload, kind)
        except (OSError, ValueError, KeyError):
            return self.send(502, {'error': 'The host GPU app is offline. Ask the owner to start the studio.'})

    def do_POST(self):
        if self.path == '/login':
            return self.login()
        if not self.authenticated():
            return
        origin = self.headers.get('Origin')
        if origin and origin not in [f'https://{self.headers.get("Host")}', f'http://{self.headers.get("Host")}']:
            return self.send(403, {'error': 'Cross-origin requests are disabled.'})
        if self.headers.get('X-Local-Token') != self.server.nonce:
            return self.send(403, {'error': 'Refresh this page before reconstructing.'})
        if self.path != '/api/inpaint':
            return self.send(404, {'error': 'Not found.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send(415, {'error': 'Expected JSON.'})
        if self.headers.get('Transfer-Encoding'):
            return self.send(400, {'error': 'Unsupported request encoding.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16_000_000:
                return self.send(413, {'error': 'Upload is too large.'})
            payload = self.rfile.read(length)
            if len(payload) != length:
                return self.send(400, {'error': 'Incomplete upload.'})
            status, data, kind = self.upstream('POST', '/api/inpaint', payload)
            if status == 202:
                job_id = json.loads(data)['id']
                if not re.fullmatch('[a-f0-9]{32}', job_id):
                    raise ValueError('Unexpected run identifier')
                with self.server.lock:
                    self.server.jobs.add(job_id)
            return self.send(status, data, kind)
        except (OSError, ValueError, KeyError):
            return self.send(502, {'error': 'The GPU app could not accept this request.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--upstream-port', type=int, default=8765)
    parser.add_argument('--credentials', type=Path, required=True)
    args = parser.parse_args()
    password = json.loads(args.credentials.read_text(encoding='utf-8'))['password']
    if len(password) < 24:
        raise ValueError('Use a randomly generated password of at least 24 characters.')
    with Gateway(('127.0.0.1', args.port), args.upstream_port, password) as server:
        print(f'Private gateway ready on loopback port {args.port}.', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
