"""Separate post-fix runtime checks; retains the completed audit and its failures."""
import base64, io, json, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from unfamiliar_studio_audit_v2 import cases
from webapp.geometry import fit_evidence, resize_binary_mask


def request(path, data=None, token=None):
    req = Request('http://127.0.0.1:8765' + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json', **({'X-Local-Token': token} if token else {})})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def main():
    out = ROOT / 'outputs/studio_fixes_v2'
    out.mkdir(exist_ok=False)
    inputs = {c['case_id']: c for c in cases()}
    planned = [('runtime_retry_1024', '3182_removal', 'reference', 'painted', 'detailed'),
        ('same_default_512', '3182_removal', 'reference', 'painted', 'standard')]
    planned += [(model + '_' + mode, '1037_removal', model, mode, 'standard')
        for model in ['lama', 'resshift'] for mode in ['expand', 'learned']]
    planned += [('thin_mask', '1037_removal', 'resshift', 'painted', 'standard'),
        ('portrait_api', '3182_removal', 'reference', 'painted', 'standard')]
    rows = []
    for name, case, model, mode, detail in planned:
        c = inputs[case]
        assert sha(c['observed']) == c['observed_sha256'] and sha(c['mask']) == c['mask_sha256']
        source = Image.open(c['observed']).convert('RGB')
        mask = Image.open(c['mask']).convert('L')
        if name == 'thin_mask':
            # Scratch wholly missed by historical 512->256 nearest-neighbour sampling.
            a = np.zeros((512,512), np.uint8); a[100:400,200] = 255
            mask = Image.fromarray(a)
        if name == 'portrait_api':
            source = source.crop((56, 0, 456, 512)); mask = mask.crop((56, 0, 456, 512))
        payload = dict(image=png(source), mask=png(mask), backbone=model, mode=mode,
            detail=detail, blend='poisson', neutralize=False)
        if model == 'reference':
            for r in c['references']: assert sha(r['path']) == r['sha256']
            payload['references'] = [png(Image.open(r['path']).convert('RGB')) for r in c['references']]
        session = request('/api/session')
        if session['busy']: raise RuntimeError('GPU occupied; no job interrupted.')
        row = dict(check=name, case_id=case, backbone=model, mode=mode, detail=detail, status='failed')
        try:
            job = request('/api/inpaint', payload, session['token']); row['job_id'] = job['id']
            started = time.monotonic()
            while True:
                result = request('/api/jobs/' + job['id'])
                if result['status'] in ['complete', 'error']: break
                if time.monotonic() - started > 1900: raise TimeoutError(job['id'])
                time.sleep(1)
            if result['status'] != 'complete': raise RuntimeError(result['message'])
            folder = ROOT / 'outputs/webapp_runs' / job['id']
            output = Image.open(folder/'result.png').convert('RGB')
            displayed = Image.open(folder/'input.png').convert('RGB')
            effective = np.asarray(Image.open(folder/'effective_mask.png')) >= 128
            metadata = json.loads((folder/'metadata.json').read_text())
            a, b = np.asarray(output), np.asarray(displayed)
            row.update(status='complete', output_sha256=sha(folder/'result.png'),
                known_pixels_unchanged=bool(np.array_equal(a[~effective], b[~effective])),
                effective_mask_pixels=int(effective.sum()), generated_masked_change=float(np.abs(a.astype(float)-b).mean(2)[effective].mean()),
                metadata_sha256=sha(folder/'metadata.json'), seconds=result['seconds'])
            assert row['known_pixels_unchanged']
            if model == 'reference':
                row['bounded_opencv_logged'] = '1 worker thread' in (folder/'inference.log').read_text(errors='replace')
                assert row['bounded_opencv_logged']
            if name == 'same_default_512':
                old = json.loads((ROOT/'outputs/unfamiliar_studio_audit_v2/rows/3182_removal__reference_512.json').read_text())
                row['byte_matches_pre_fix_default'] = row['output_sha256'] == old['output_sha256']
            if name == 'thin_mask':
                native = resize_binary_mask(mask, (256,256))
                row['native_mask_pixels'] = int((np.asarray(native)>=128).sum())
                assert row['native_mask_pixels'] > 0 and row['effective_mask_pixels'] == 300
            if name == 'portrait_api':
                expected, _, _, geometry = fit_evidence(source, mask, Image.new('L', source.size, 255))
                row['aspect_preserving_input_exact'] = np.array_equal(np.asarray(expected), b)
                row['upload_geometry'] = metadata['upload_geometry']
                assert row['aspect_preserving_input_exact']
        except Exception as error:
            row.update(status='failed', error=str(error))
        rows.append(row); write_new(out/(name+'.json'), row)
        print(name, row['status'], flush=True)
    write_new(ROOT/'research/studio_fixes_verification_v2.json', dict(
        scope='Separate post-fix functional runs on now-observed development images. No accuracy claim or replacement of original audit failures.',
        server_sha256=sha(ROOT/'webapp/server.py'), worker_sha256=sha(ROOT/'scripts/studio_inference_worker.py'),
        rows=rows, final_test_used=False))


if __name__ == '__main__': main()
