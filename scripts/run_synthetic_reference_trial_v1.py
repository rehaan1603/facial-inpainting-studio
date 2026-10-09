"""Matched generator-only diagnostic; no clean targets or target metrics accessed."""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reference_inpaint import reconstruct
from reference_blending import harmonize_reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/synthetic_references_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    protocol = json.loads((OUT/'protocol.json').read_text())
    # Model comparison settings are fixed before the first generated output.
    runtime = {}
    rows = []
    records = OUT/'runs'
    records.mkdir(exist_ok=False)
    for case in protocol['cases']:
        for item in [case['input'], case['mask'], *case['references']]:
            assert sha(item['path']) == item['sha256']
        observed = np.asarray(Image.open(case['input']['path']).convert('RGB'))
        mask = np.asarray(Image.open(case['mask']['path']).convert('L')) >= 128
        conditioning = observed.copy()
        conditioning[mask] = 127
        parent = records/case['case_id']
        parent.mkdir()
        conditioning_path = parent/'conditioning.png'
        Image.fromarray(conditioning).save(conditioning_path)
        for arm in protocol['arms']:
            folder = parent/arm
            folder.mkdir()
            refs = case['references'][:1] if arm == 'first_synthetic' else case['references']
            scale = 0. if arm == 'reference_off' else .8
            row = dict(case_id=case['case_id'], arm=arm, status='failed',
                       reference_count=len(refs), scale=scale, reference_hashes=[r['sha256'] for r in refs])
            start = time.monotonic()
            try:
                metadata = reconstruct(conditioning_path, case['mask']['path'], [r['path'] for r in refs],
                    folder/'prediction.png', seed=17, steps=30, scale=scale, strength=1.,
                    model_resolution=512, blend='hard', runtime=runtime, progress_path=folder/'progress.json')
                raw = np.asarray(Image.open(folder/'prediction_raw.png').convert('RGB'))
                result, blend = harmonize_reference(raw, observed, mask)
                assert np.array_equal(result[~mask], observed[~mask])
                Image.fromarray(result).save(folder/'result.png')
                row.update(status='complete', result=str((folder/'result.png').relative_to(OUT)),
                           result_sha256=sha(folder/'result.png'), conditioning_sha256=sha(conditioning_path),
                           outside_mask_exact=True, seconds=time.monotonic()-start, blending=blend)
            except Exception as error:
                row['error'] = str(error)
            rows.append(row)
            (folder/'row.json').write_text(json.dumps(row, indent=2))
            (OUT/'generation.json').write_text(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),
                 'runner_sha256':sha(__file__), 'rows':rows}, indent=2))
            print('TRIAL RESULT',case['case_id'],arm,row['status'],flush=True)
    print('TRIAL COMPLETE',len(rows),flush=True)


if __name__ == '__main__':
    main()
