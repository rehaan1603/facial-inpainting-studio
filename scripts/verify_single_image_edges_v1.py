"""Matched raw/finished studio checks on new and existing development cases."""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_human_faces_baselines import request, checked_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/single_image_edges_v1'
PACK = Path.home() / 'Downloads/Human dataset/Ready to test - Human Faces v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    new = json.loads((PACK / 'inference_manifest.json').read_text())['cases']
    plan = [(c, model, 'human_faces') for c in new if c['condition'] != 'blur_noise'
            for model in ['lama', 'resshift']]
    old_base = ROOT / 'outputs/unfamiliar_studio_v2'
    old = json.loads((old_base / 'inference_manifest.json').read_text())['cases']
    for c in old:
        if c['case_id'] not in ['3182_removal', '1037_removal']:
            continue
        plan.append((dict(case_id=c['case_id'],
                          input={'path': c['observed'], 'sha256': c['observed_sha256']},
                          mask={'path': c['mask'], 'sha256': c['mask_sha256']}),
                     'resshift', 'existing_development'))
    assert len(plan) == 18
    saved = OUT/'generation.json'
    rows = json.loads(saved.read_text())['rows'] if saved.exists() else []
    planned = {(c['case_id'], model, dataset) for c, model, dataset in plan}
    done = set()
    for row in rows:
        key = (row['case_id'], row['model'], row['dataset'])
        assert key in planned and key not in done
        done.add(key)
        if row['status'] == 'complete':
            folder = ROOT/'outputs/webapp_runs'/row['job_id']
            assert digest(folder/'result_raw.png') == row['raw_sha256']
            assert digest(folder/'result.png') == row['finished_sha256']
    for c, model, dataset in plan:
        if (c['case_id'], model, dataset) in done:
            continue
        session = request('/api/session')
        if session['busy']:
            print('Studio occupied; saving completed checks. Resume later without replacing them.', flush=True)
            break
        row = dict(case_id=c['case_id'], model=model, dataset=dataset, status='failed')
        try:
            payload = dict(backbone=model, mode='painted', detail='standard', blend='poisson')
            for field, name in [('image', 'input'), ('mask', 'mask')]:
                payload[field] = 'data:image/png;base64,' + base64.b64encode(checked_file(c[name]).read_bytes()).decode()
            job = request('/api/inpaint', payload, session['token'])
            row['job_id'] = job['id']
            start = time.monotonic()
            while True:
                result = request('/api/jobs/' + job['id'])
                if result['status'] in ['complete', 'error']:
                    break
                if time.monotonic() - start > 600:
                    raise TimeoutError('Job timeout; stop further submissions.')
                time.sleep(1)
            if result['status'] != 'complete':
                raise RuntimeError(result['message'])
            folder = ROOT / 'outputs/webapp_runs' / job['id']
            metadata = json.loads((folder / 'metadata.json').read_text())
            assert metadata['blending']['mode'] == 'poisson'
            row.update(status='complete', raw_sha256=digest(folder/'result_raw.png'),
                       finished_sha256=digest(folder/'result.png'), blending=metadata['blending'])
        except Exception as error:
            row['error'] = str(error)
        rows.append(row)
        (OUT/'generation.json').write_text(json.dumps({'rows': rows}, indent=2))
        print(c['case_id'], model, row['status'], flush=True)
        if row['status'] != 'complete':
            break
    # Separate evaluation: no clean target was provided during generation.
    targets = {c['case_id']: {'path': c['target'], 'sha256': c['target_sha256']}
               for c in json.loads((PACK/'evaluation_manifest.json').read_text())['cases']}
    targets.update({c['case_id']: c['target'] for c in json.loads(
        (old_base/'evaluation_manifest.json').read_text())['cases']})
    previous = {(r['case_id'], r['model']): r for r in json.loads(
        (ROOT/'outputs/human_faces_baseline_v1/scores.json').read_text())['rows']}
    for row in rows:
        if row['status'] != 'complete':
            continue
        folder = ROOT/'outputs/webapp_runs'/row['job_id']
        clean = np.asarray(Image.open(checked_file(targets[row['case_id']])).convert('RGB'))
        observed = np.asarray(Image.open(folder/'input.png').convert('RGB'))
        mask = np.asarray(Image.open(folder/'effective_mask.png').convert('L')) >= 128
        panels = [('Clean comparison', clean), ('Damaged input', observed)]
        for key, name in [('raw', 'result_raw.png'), ('finished', 'result.png')]:
            rgb = np.asarray(Image.open(folder/name).convert('RGB'))
            assert rgb.shape == clean.shape == observed.shape
            row[key+'_masked_mae'] = float(np.abs((rgb.astype(float)-clean)[mask]/255).mean())
            row[key+'_outside_exact'] = bool(np.array_equal(rgb[~mask], observed[~mask]))
            assert row[key+'_outside_exact']
            panels.append((key, rgb))
        p = previous.get((row['case_id'], row['model']))
        if p:
            row['raw_matches_previous_output'] = row['raw_sha256'] == p['output_sha256']
            assert row['raw_matches_previous_output']
        sheet = Image.new('RGB', (1024, 286), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, (label, rgb) in enumerate(panels):
            draw.text((j*256+6, 8), label, fill='black')
            sheet.paste(Image.fromarray(rgb).resize((256,256)), (j*256,30))
        sheet.save(OUT/(row['case_id']+'__'+row['model']+'.png'))
    receipt = dict(scope='Development-only boundary matching; no training or identity accuracy claim',
                   runner_sha256=digest(__file__), expected=18, attempted=len(rows), rows=rows)
    (OUT/'scores.json').write_text(json.dumps(receipt, indent=2))
    print('Saved', len(rows), 'paired raw/finished checks', flush=True)


if __name__ == '__main__':
    main()
