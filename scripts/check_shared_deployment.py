"""Check the HTTPS deployment with synthetic pixels; no dataset photos leave the laptop."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-job', help='Resume checking a previously accepted synthetic deployment test job')
    args = parser.parse_args()
    access = json.loads((ROOT / '.local-share/access.json').read_text(encoding='utf-8'))
    public = json.loads((ROOT / '.local-share/state.json').read_text(encoding='utf-8'))['url']
    auth = 'Basic ' + base64.b64encode(('studio:' + access['password']).encode()).decode()

    def request(path, data=None, authenticated=True, extra=None):
        headers = {'Authorization': auth} if authenticated else {}
        headers.update(extra or {})
        try:
            with urlopen(Request(public + path, data=data, headers=headers), timeout=60) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    login_status, login_page = request('/', authenticated=False)
    assert login_status == 200 and b'Demo password' in login_page
    assert request('/api/session', authenticated=False)[0] == 401
    status, page = request('/')
    assert status == 200 and b'Uploads travel through Cloudflare' in page and b'No cloud upload' not in page
    assert request('/api/demo?reference=1')[0] == 404
    assert request('/runs/' + 'a'*32 + '/result.png')[0] == 404
    status, body = request('/api/session')
    session = json.loads(body)
    assert status == 200 and session['shared'] and not session['sample_available']
    yy, xx = np.indices((256, 256))
    source = np.stack([xx, yy, (xx + yy)//2], axis=-1).astype('uint8')
    binary = np.zeros((256, 256), dtype=bool)
    binary[100:150, 100:150] = True
    source[binary] = 90

    def encode(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

    payload = json.dumps({'image': encode(source), 'mask': encode(binary.astype('uint8')*255),
                          'backbone': 'lama', 'mode': 'painted'}).encode()
    if args.resume_job:
        job_id = args.resume_job
    else:
        status, body = request('/api/inpaint', payload, extra={'Content-Type': 'application/json',
                               'X-Local-Token': session['token'], 'Origin': public})
        assert status == 202, (status, body)
        job_id = json.loads(body)['id']
    print('Public HTTPS upload accepted; checking real GPU reconstruction...', flush=True)
    deadline = time.monotonic() + 600
    transient_errors = 0
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            status, body = request('/api/jobs/' + job_id)
            if status in [502, 503, 504, 520, 522, 524]:
                transient_errors += 1
                continue
            job = json.loads(body)
        except (URLError, TimeoutError):
            transient_errors += 1
            continue
        assert status == 200 and job['status'] != 'error', job
        if job['status'] == 'complete':
            break
    else:
        raise RuntimeError('Public GPU run timed out')
    status, result = request(job['result'])
    assert status == 200
    output = np.asarray(Image.open(io.BytesIO(result)).convert('RGB'))
    assert output.shape == source.shape
    assert np.array_equal(output[~binary], source[~binary])
    changed = int(np.any(output != source, axis=-1)[binary].sum())
    assert changed > 0
    assert request(job['mask'])[0] == 200
    report = {'url': public, 'checks': ['anonymous_denied', 'authenticated_page', 'shared_upload_notice',
              'dataset_sample_denied', 'old_run_denied', 'https_upload', 'real_lama_gpu_inference',
              'result_and_mask_download', 'known_pixels_exact'], 'job_id': job_id,
              'inference_seconds': job['seconds'], 'changed_mask_pixels': changed,
              'transient_poll_errors': transient_errors, 'resumed_existing_job': bool(args.resume_job),
              'result_sha256': hashlib.sha256(result).hexdigest(),
              'scope': 'Deployment verification using a synthetic colour gradient, not a facial-quality evaluation.'}
    (ROOT / 'research/shared_deployment_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
