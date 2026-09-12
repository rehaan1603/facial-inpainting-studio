"""Exercise real local HTTP reference inference and inspect the downloaded PNG."""
import argparse
import io
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from atomic_records import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detail', choices=['standard', 'detailed'], default='standard')
    parser.add_argument('--references', type=int, choices=[3, 4], default=3)
    args = parser.parse_args()
    base = 'http://127.0.0.1:8765'
    def fetch(path, data=None, token=None):
        request = urllib.request.Request(base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={'Content-Type': 'application/json', 'X-Local-Token': token or ''})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    session = json.loads(fetch('/api/session'))
    if session['busy']:
        raise RuntimeError('A reconstruction is already running.')
    demo = json.loads(fetch('/api/demo?reference=1'))
    demo.update(backbone='reference', mode='painted', blend='poisson', detail=args.detail)
    demo['references'] = demo['references'][:args.references]
    start = time.monotonic()
    job = json.loads(fetch('/api/inpaint', demo, session['token']))
    while True:
        status = json.loads(fetch('/api/jobs/' + job['id']))
        if status['status'] == 'error':
            raise RuntimeError(status['message'])
        if status['status'] == 'complete':
            break
        if time.monotonic() - start > 900:
            raise TimeoutError('Reference HTTP inference did not finish in 15 minutes.')
        time.sleep(.5)
    def pixels(path):
        with Image.open(io.BytesIO(fetch(path))) as image:
            return np.asarray(image).copy()
    observed, result, mask = pixels(status['input']), pixels(status['result']), pixels(status['mask']) >= 128
    metadata = json.loads(fetch(status['metadata']))
    assert result.shape == observed.shape == (512, 512, 3)
    assert mask.shape == (512, 512) and mask.any()
    assert np.array_equal(result[~mask], observed[~mask])
    changed = int(np.any(result != observed, axis=2).sum())
    assert changed > 0
    assert metadata['reference_count'] == args.references
    assert metadata['strength'] == .99
    assert metadata['model_resolution'] == ([1024, 1024] if args.detail == 'detailed' else [512, 512])
    record = dict(job_id=job['id'], detail=args.detail, references=args.references,
                  changed_pixels=changed, outside_mask_exact=True,
                  wall_seconds=time.monotonic()-start, inference_seconds=status['seconds'],
                  model_resolution=metadata['model_resolution'], result_sha256=metadata['result_sha256'],
                  environment=metadata['environment'])
    path = ROOT / 'research/reference_webapp_repair_checks.json'
    report = json.loads(path.read_text()) if path.exists() else {
        'scope': 'Real HTTP/GPU/PNG functional checks on the identity_916 engineering sample; no identity-fidelity claim.', 'runs': []}
    report['runs'].append(record)
    write_json(path, report)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
