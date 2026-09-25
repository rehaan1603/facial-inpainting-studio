"""Run a frozen context-support screen. This process never reads clean truth/gallery."""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_support import probes, calibration_input, probe_loss, choose_radius, compose
from webapp.geometry import resize_binary_mask
from inpaint import Inpainter

BASE = ROOT / 'outputs/context_support_v1'
PROTOCOL = ROOT / 'research/protocols/context_support_v1.json'
MANIFEST = ROOT / 'outputs/distortion_aware_v1/inference_manifest.json'


def read_checked(path, expected, mode='RGB'):
    if sha(path) != expected:
        raise ValueError('Input hash mismatch')
    with Image.open(path) as image:
        return image.convert(mode).copy()


def main():
    cfg = json.loads(PROTOCOL.read_text())
    all_cases = json.loads(MANIFEST.read_text())['cases']
    cases = [next(c for c in all_cases if c['case_id'] == cid) for cid in cfg['cases']]
    assert not (set(c['identity'] for c in cases) & set(cfg['reserved_identities']))
    signature = {'protocol_sha256': sha(PROTOCOL), 'inference_manifest_sha256': sha(MANIFEST),
                 'sources': {p: sha(ROOT / p) for p in [
                     'scripts/run_context_support.py', 'src/preservation/context_support.py',
                     'scripts/inpaint.py', 'scripts/resshift_adapter.py', 'webapp/geometry.py']}}
    lock = BASE / 'signature.json'
    if lock.exists():
        assert json.loads(lock.read_text())['signature'] == signature, 'Frozen source/input changed'
    else:
        write_new(lock, {'signature': signature, 'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    if (BASE / 'comparison.json').exists():
        raise FileExistsError('Completed run is immutable')
    torch.set_num_threads(4)
    model = Inpainter(cfg['backbone'])
    rows, selections, ledger = [], [], []

    def generate(key, observed, mask, seed, phase):
        dest = BASE / 'generations' / (key + '.json')
        if dest.exists():
            receipt = json.loads(dest.read_text())
            if receipt['status'] == 'complete':
                assert sha(receipt['output']) == receipt['output_sha256']
        else:
            started = time.monotonic()
            receipt = {'key': key, 'phase': phase, 'seed': seed,
                       'input_sha256': hashlib.sha256(observed.tobytes()).hexdigest(),
                       'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(), 'status': 'failed'}
            try:
                output = model(observed, mask, seed)
                rgb = np.rint(np.clip(output, 0, 1) * 255).astype('uint8')
                assert np.isfinite(output).all()
                path = dest.with_suffix('.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError('Orphan output; preserve and investigate instead of overwriting')
                Image.fromarray(rgb).save(path)
                receipt.update(status='complete', output=str(path), output_sha256=sha(path))
            except Exception as error:
                receipt['error'] = f'{type(error).__name__}: {error}'
            receipt['seconds'] = time.monotonic() - started
            write_new(dest, receipt)
        ledger.append(receipt)
        print(f"GEN {len(ledger)}/104 {key} {receipt['status']}", flush=True)
        return np.asarray(read_checked(receipt['output'], receipt['output_sha256'])) if receipt['status'] == 'complete' else None

    for case in cases:
        original = read_checked(case['observed'], case['observed_sha256'])
        original_mask_image = read_checked(case['mask'], case['mask_sha256'], 'L')
        assert original.size == original_mask_image.size == (512, 512)
        original_rgb = np.asarray(original)
        original_mask = np.asarray(original_mask_image) >= 128
        native = np.asarray(original.resize((256, 256), Image.Resampling.LANCZOS)).astype('float32') / 255
        missing = np.asarray(resize_binary_mask(original_mask_image, (256, 256))) >= 128
        hidden, info = probes(missing, cfg['radii'], **cfg['probe'])
        probe_path = BASE / case['case_id'] / 'probe_mask.png'
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        if not probe_path.exists():
            Image.fromarray(hidden.astype('uint8') * 255).save(probe_path)
        for seed in cfg['seeds']:
            prefix = f"{case['case_id']}_s{seed}"
            selection_path = BASE / 'selections' / (prefix + '.json')
            losses = {}
            for radius in cfg['radii']:
                x, m = calibration_input(native, missing, hidden, radius)
                rgb = generate(f'{prefix}_probe_r{radius}', x, m, seed, 'probe')
                if rgb is not None and info['eligible']:
                    losses[radius] = probe_loss(rgb.astype('float32') / 255, native, hidden)
            selected, status = choose_radius(losses, cfg['radii'], info['eligible'])
            selection = {'case_id': case['case_id'], 'identity': case['identity'], 'seed': seed,
                         'chosen_radius': selected, 'status': status, 'probe_info': info,
                         'probe_mask_sha256': sha(probe_path), 'losses': {str(k): v for k, v in losses.items()}}
            # Lock selection BEFORE the final candidate bank is generated or inspected.
            if selection_path.exists():
                assert json.loads(selection_path.read_text()) == json.loads(json.dumps(selection))
            else:
                write_new(selection_path, selection)
            selections.append(selection)
            finals = {}
            for radius in cfg['radii']:
                x, m = calibration_input(native, missing, np.zeros_like(hidden), radius)
                finals[radius] = generate(f'{prefix}_final_r{radius}', x, m, seed, 'final')
            seeds = [finals[0]]
            for offset in cfg['extra_seeds']:
                x, m = calibration_input(native, missing, np.zeros_like(hidden), 0)
                seeds.append(generate(f'{prefix}_extra_s{seed+offset}', x, m, seed+offset, 'seed_control'))
            x, m = calibration_input(native, missing, np.zeros_like(hidden), 4)
            extra_support = generate(f'{prefix}_extra_r4', x, m, seed+10000, 'support_control')
            def mean(bank):
                return np.mean(bank, axis=0) if all(p is not None for p in bank) else None
            random_index = int(hashlib.sha256(prefix.encode()).hexdigest()[:8], 16) % len(cfg['radii'])
            outputs = {'observed': original_rgb, **{f'fixed_r{r}': p for r, p in finals.items()},
                       'context_selected': finals[selected], 'random_support': finals[cfg['radii'][random_index]],
                       'support_mean4': mean(list(finals.values())),
                       'support_mean5': mean(list(finals.values()) + [extra_support]), 'seed_mean5': mean(seeds)}
            for arm in cfg['arms']:
                key = prefix + '_' + arm
                row = {'key': key, 'case_id': case['case_id'], 'identity': case['identity'],
                       'seed': seed, 'mode': arm, 'status': 'failed'}
                prediction = outputs[arm]
                if prediction is not None:
                    if arm == 'observed':
                        output = original_rgb.copy()
                    else:
                        up = Image.fromarray(np.rint(np.clip(prediction, 0, 255)).astype('uint8')).resize(original.size, Image.Resampling.LANCZOS)
                        output = compose(original_rgb, np.asarray(up), original_mask)
                    path = BASE / 'outputs' / (key + '.png')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        assert np.array_equal(np.asarray(Image.open(path)), output)
                    else:
                        Image.fromarray(output).save(path)
                    row.update(status='complete', output=str(path), output_sha256=sha(path),
                               known_pixels_unchanged=bool(np.array_equal(output[~original_mask], original_rgb[~original_mask])))
                else:
                    row['error'] = 'Required generation failed; see immutable generation ledger'
                rows.append(row)
    write_new(BASE / 'comparison.json', {'signature_sha256': sha(lock), 'rows': rows,
              'selections': selections, 'generation_ledger': ledger, 'final_test_used': False})
    print(json.dumps({'generations': len(ledger), 'rows': len(rows), 'complete': sum(r['status']=='complete' for r in rows)}))


if __name__ == '__main__':
    main()
