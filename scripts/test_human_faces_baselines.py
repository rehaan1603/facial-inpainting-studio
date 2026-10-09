"""Run existing studio baselines; retain every attempt, then score separately."""
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def request(path, data=None, token=None):
    req = Request('http://127.0.0.1:8765' + path,
                  data=json.dumps(data).encode() if data is not None else None,
                  headers={'Content-Type': 'application/json',
                           **({'X-Local-Token': token} if token else {})})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def checked_file(item):
    p = Path(item['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Input hash mismatch')
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pack / 'inference_manifest.json').read_text())
    cases = [c for c in manifest['cases'] if c['case_id'].startswith(('photo_01_', 'photo_02_'))
             and c['condition'] != 'blur_noise']
    if len(cases) != 4:
        raise ValueError('Expected four missing-region cases.')
    if request('/api/session')['busy']:
        raise RuntimeError('Studio is occupied; no job interrupted.')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        for model in ['lama', 'resshift']:
            row = {'case_id': case['case_id'], 'model': model, 'status': 'failed'}
            try:
                session = request('/api/session')
                if session['busy']:
                    raise RuntimeError('Studio occupied; stopping without interrupting user work.')
                payload = {'backbone': model, 'mode': 'painted', 'detail': 'standard'}
                for field, name in [('image', 'input'), ('mask', 'mask')]:
                    payload[field] = 'data:image/png;base64,' + base64.b64encode(
                        checked_file(case[name]).read_bytes()).decode()
                job = request('/api/inpaint', payload, session['token'])
                row['job_id'] = job['id']
                start = time.monotonic()
                while True:
                    result = request('/api/jobs/' + job['id'])
                    if result['status'] in ['complete', 'error']:
                        break
                    if time.monotonic() - start > 600:
                        raise TimeoutError('GPU job still running; stop further submissions.')
                    time.sleep(1)
                if result['status'] != 'complete':
                    raise RuntimeError(result['message'])
                output = ROOT / 'outputs/webapp_runs' / job['id'] / Path(result['result']).name
                saved = args.output / f"{case['case_id']}__{model}.png"
                saved.write_bytes(output.read_bytes())
                row.update(status='complete', output=saved.name,
                           output_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                           seconds=result['seconds'])
            except Exception as error:
                row['error'] = str(error)
            rows.append(row)
            (args.output / 'generation.json').write_text(json.dumps({'rows': rows}, indent=2))
            print(row['case_id'], model, row['status'], flush=True)
            if row['status'] != 'complete':
                break
        if rows[-1]['status'] != 'complete':
            break
    # Clean targets are opened only after the generation phase.
    targets = {c['case_id']: c for c in json.loads(
        (args.pack / 'evaluation_manifest.json').read_text())['cases']}
    for case in cases:
        e = targets[case['case_id']]
        target = np.asarray(Image.open(checked_file(
            {'path': e['target'], 'sha256': e['target_sha256']})).convert('RGB'))
        observed = np.asarray(Image.open(checked_file(case['input'])).convert('RGB'))
        mask = np.asarray(Image.open(checked_file(case['mask'])).convert('L')) >= 128
        panels = [('Clean comparison', target), ('Damaged input', observed)]
        for row in rows:
            if row['case_id'] != case['case_id'] or row['status'] != 'complete':
                continue
            rgb = np.asarray(Image.open(args.output / row['output']).convert('RGB'))
            delta = (rgb.astype(float) - target.astype(float)) / 255
            row['masked_mae_0_1'] = float(np.abs(delta[mask]).mean())
            row['masked_psnr_db'] = float(-10 * np.log10(max(np.mean(delta[mask]**2), 1e-12)))
            row['outside_mask_exact'] = bool(np.array_equal(rgb[~mask], observed[~mask]))
            row['input_masked_mae_0_1'] = float(np.abs(
                (observed.astype(float) - target.astype(float))[mask] / 255).mean())
            panels.append((row['model'], rgb))
        sheet = Image.new('RGB', (256 * len(panels), 286), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (label, rgb) in enumerate(panels):
            draw.text((i * 256 + 8, 8), label, fill='black')
            sheet.paste(Image.fromarray(rgb).resize((256, 256)), (i * 256, 30))
        sheet.save(args.output / (case['case_id'] + '_comparison.png'))
    receipt = {'scope': 'Development smoke only; two photos, no verified identities, source/provenance unknown',
               'expected_attempts': 8, 'actual_attempts': len(rows),
               'inference_manifest_sha256': hashlib.sha256((args.pack / 'inference_manifest.json').read_bytes()).hexdigest(),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'training_started': False, 'rows': rows}
    (args.output / 'scores.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
