"""Build a local development pack, never training or assigning identity labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        raise ValueError('Use 1–20 development photographs.')
    audit = ROOT / 'outputs/human_faces_intake_20261001'
    rows = json.loads((audit / 'manifest.json').read_text())
    real = [r for r in rows if r['group'] == 'Real Images']
    selected = sorted(real, key=lambda r: hashlib.sha256(
        ('human_faces_development_v1:' + r['sha256']).encode()).hexdigest())[:args.count]
    for row in selected:
        if sha(args.source / row['file']) != row['sha256']:
            raise ValueError('Source differs from audited image.')
    args.output.mkdir(parents=True, exist_ok=False)
    cases, targets = [], []
    for index, row in enumerate(selected, 1):
        folder = args.output / f'Photo {index:02d}'
        folder.mkdir()
        with Image.open(args.source / row['file']) as im:
            # Explicitly interpolated low-resolution source, not new HQ ground truth.
            target = ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),
                                  (512, 512), method=Image.Resampling.BICUBIC,
                                  color=(128, 128, 128))
        target.save(folder / 'Comparison only - original.png')
        clean = np.asarray(target)
        for condition in ['central_missing', 'thin_scratches', 'blur_noise']:
            dest = folder / condition
            dest.mkdir()
            mask = np.zeros((512, 512), dtype=np.uint8)
            if condition == 'central_missing':
                mask[205:295, 165:347] = 255
            elif condition == 'thin_scratches':
                mask[145:350, 250:258] = 255
                mask[250:258, 150:360] = 255
            else:
                mask[100:420, 95:417] = 255
            observed = clean.copy()
            selected_pixels = mask > 0
            seed = 20261001 + index
            if condition == 'blur_noise':
                blurred = np.asarray(target.filter(ImageFilter.GaussianBlur(3))).astype(float)
                noise = np.random.default_rng(seed).normal(0, 8, clean.shape)
                corrupted = np.rint(np.clip(blurred + noise, 0, 255)).astype('uint8')
                observed[selected_pixels] = corrupted[selected_pixels]
                confidence = np.where(selected_pixels, 128, 255).astype('uint8')
            else:
                observed[selected_pixels] = 0
                confidence = 255 - mask
            assert np.array_equal(observed[~selected_pixels], clean[~selected_pixels])
            assert np.any(observed[selected_pixels] != clean[selected_pixels])
            paths = {}
            for name, pixels in [('input', observed), ('mask', mask), ('evidence', confidence)]:
                p = dest / (name + '.png')
                Image.fromarray(pixels).save(p)
                paths[name] = {'path': str(p.resolve()), 'sha256': sha(p)}
            case_id = f'photo_{index:02d}_{condition}'
            cases.append({'case_id': case_id, 'condition': condition,
                          'seed': seed, 'references': [], **paths})
            target_path = folder / 'Comparison only - original.png'
            targets.append({'case_id': case_id, 'target': str(target_path.resolve()),
                            'target_sha256': sha(target_path),
                            'source_file': row['file'], 'source_sha256': row['sha256']})
    scope = {'scope': 'Local development only; provenance and identity overlap unverified',
             'identity_labels': False, 'training_started': False,
             'source_resolution': [178, 218], 'canvas_resolution': [512, 512],
             'preprocessing': 'Aspect-preserving bicubic fit with gray padding; no new detail',
             'selection': 'Fixed SHA256 order, no quality/result selection',
             'script_sha256': sha(__file__)}
    (args.output / 'inference_manifest.json').write_text(json.dumps(
        {**scope, 'cases': cases}, indent=2), encoding='utf-8')
    (args.output / 'evaluation_manifest.json').write_text(json.dumps(
        {**scope, 'cases': targets}, indent=2), encoding='utf-8')
    (args.output / 'START HERE.txt').write_text('''LOCAL HUMAN FACES DEVELOPMENT TESTS

Photo folders are individual photographs, NOT verified person/identity groups.
The original dataset has not been changed. These are low-resolution source
photos interpolated onto a 512px canvas, not high-resolution originals.

MISSING AREA TEST: open http://127.0.0.1:8765/ ; select LaMa or ResShift;
upload input.png and mask.png from central_missing or thin_scratches.
Choose Use mask exactly. White mask means replace, black means preserve.

BLUR/NOISE: the separate blur_noise folder contains input, mask and evidence.
Evidence: white reliable, gray partially damaged. A genuine same-person
reference restoration run is NOT prepared: verified reference photos are absent.
Do not treat the original or unrelated generated images as reference photos.

Compare output with Comparison only - original.png. NEVER upload that clean
comparison as a reference or use it to generate synthetic test references.
Inference manifests exclude clean targets; evaluation manifests contain them.

Synthetic-angle experiment is pending. It must use damaged input only, with
fixed seeds and all failures retained. Synthetic views are not real observations.
No training, identity grouping, independence or publication-quality claims.
''', encoding='utf-8')
    print(json.dumps({'photographs': len(selected), 'cases': len(cases),
                      'output': str(args.output), 'training_started': False}))


if __name__ == '__main__':
    main()
