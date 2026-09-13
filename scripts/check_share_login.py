"""Verify browser-style HTTPS form login without logging credentials or cookies."""
import json
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT / '.local-share/state.json').read_text(encoding='utf-8'))
access = json.loads((ROOT / '.local-share/access.json').read_text(encoding='utf-8'))
url = state['url']
with urlopen(url, timeout=30) as response:
    assert response.status == 200 and b'Demo password' in response.read()
jar = CookieJar()
browser = build_opener(HTTPCookieProcessor(jar))
request = Request(url + '/login', data=urlencode({'password': access['password']}).encode(),
                  headers={'Content-Type': 'application/x-www-form-urlencoded', 'Origin': url})
with browser.open(request, timeout=30) as response:
    assert response.status == 200 and b'Reconstruct a face.' in response.read()
with browser.open(url + '/api/session', timeout=30) as response:
    session = json.load(response)
    assert session['shared'] and not session['sample_available']
assert any(cookie.secure and cookie.name == 'studio_session' for cookie in jar)
try:
    urlopen(url + '/api/session', timeout=30)
    raise AssertionError('Anonymous access was unexpectedly allowed')
except HTTPError as error:
    assert error.code == 401
report = {'url': url, 'checks': ['public_login_page', 'form_login_redirect_to_studio',
          'secure_session_cookie', 'authenticated_api_access', 'anonymous_api_denied'],
          'browser_observation': 'In-app browser visibly displays the normal sign-in form instead of ERR_INVALID_AUTH_CREDENTIALS.'}
(ROOT / 'research/shared_login_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('HTTPS form login, cookie authentication and protected API checks passed.')
