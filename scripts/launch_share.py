"""Start a temporary HTTPS demo backed by the existing local GPU studio."""
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.local-share'
URL = 'https://github.com/cloudflare/cloudflared/releases/download/2026.9.1/cloudflared-windows-amd64.exe'
SHA = '2837888cc0f5d58f15b6dc478376de90b4d3ba5241c7947455d1e0a0df429712'


def main():
    if os.name != 'nt':
        raise SystemExit('This launcher is for the recorded Windows laptop environment.')
    PRIVATE.mkdir(exist_ok=True)
    credentials = PRIVATE / 'access.json'
    if not credentials.exists():
        credentials.write_text(json.dumps({'username': 'studio', 'password': secrets.token_urlsafe(24)}), encoding='utf-8')
    access = json.loads(credentials.read_text(encoding='utf-8'))
    authorization = 'Basic ' + base64.b64encode(('studio:' + access['password']).encode()).decode()
    state_path = PRIVATE / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        try:
            with urlopen(Request(state['url'] + '/api/session', headers={'Authorization': authorization}), timeout=10) as response:
                if json.load(response).get('shared'):
                    print('Existing private demo: ' + state['url'])
                    return
        except (OSError, ValueError):
            raise SystemExit('Previous sharing session is unavailable. Run Stop Sharing.ps1, then start again.')
    for config in ['config.yml', 'config.yaml']:
        if (Path.home() / '.cloudflared' / config).exists():
            raise SystemExit('An existing Cloudflare configuration may conflict with Quick Tunnels; it was not changed.')
    binary = PRIVATE / 'cloudflared.exe'
    if not binary.exists():
        print('Downloading the pinned official Cloudflare tunnel client...', flush=True)
        with urlopen(URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA:
            raise SystemExit('Tunnel client checksum mismatch.')
        binary.write_bytes(data)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
        raise SystemExit('Tunnel client checksum mismatch.')
    subprocess.run([sys.executable, str(ROOT / 'scripts/launch_studio.py'), '--no-browser'], cwd=ROOT, check=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    children = []
    try:
        with (PRIVATE / 'gateway.log').open('ab', buffering=0) as log:
            gateway = subprocess.Popen([sys.executable, str(ROOT / 'webapp/share_gateway.py'), '--credentials', str(credentials)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags)
        children.append(gateway)
        for _ in range(20):
            if gateway.poll() is not None:
                raise RuntimeError('Sharing gateway could not start. See .local-share/gateway.log.')
            try:
                with urlopen(Request('http://127.0.0.1:8877/api/session', headers={'Authorization': authorization}), timeout=2) as response:
                    if json.load(response).get('shared'):
                        break
            except (OSError, ValueError):
                time.sleep(.5)
        else:
            raise RuntimeError('Sharing gateway did not become ready.')
        logfile = PRIVATE / 'tunnel.log'
        with logfile.open('wb', buffering=0) as log:
            tunnel = subprocess.Popen([str(binary), 'tunnel', '--no-autoupdate', '--protocol', 'http2', '--url', 'http://127.0.0.1:8877'],
                                      cwd=PRIVATE, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags)
        children.append(tunnel)
        for _ in range(100):
            if tunnel.poll() is not None:
                raise RuntimeError('Tunnel exited. See .local-share/tunnel.log.')
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', logfile.read_text(encoding='utf-8', errors='replace'))
            if match:
                public_url = match[0]
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Cloudflare did not issue a URL. See .local-share/tunnel.log.')
        state = {'url': public_url, 'gateway_pid': gateway.pid, 'tunnel_pid': tunnel.pid,
                 'scope': 'Temporary laptop-backed private research demonstration'}
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        (PRIVATE / 'LOGIN.txt').write_text('Temporary demo: ' + public_url + '\nUsername: studio\nPassword: ' + access['password'] +
                                         '\n\nKeep this file private. The laptop must remain awake and connected.\n', encoding='utf-8')
        print('Private demo URL: ' + public_url)
        print('Your username and password are in .local-share/LOGIN.txt (excluded from GitHub).')
    except Exception:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        raise


if __name__ == '__main__':
    main()
