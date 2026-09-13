"""Restore frozen reference inputs from user-downloaded HQ photos, without selection."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plan(hq):
    demo_dir = ROOT / 'outputs/reference_examples/identity_916'
    demo = json.loads((demo_dir / 'manifest.json').read_text(encoding='utf-8'))
    diagnostic = json.loads((ROOT / 'outputs/reference_diagnostics_v1/manifest.json').read_text(encoding='utf-8'))
    cases = [(demo_dir, demo['images'], demo['mask']['polygon'], demo['observed']['fill_rgb'],
              demo['mask']['sha256'], demo['observed']['sha256'])]
    for case in diagnostic['cases']:
        cases.append((ROOT / Path(case['mask']).parent, case['images'],
                      case['mask_geometry']['polygon'], case['mask_geometry']['fill_rgb'],
                      case['mask_sha256'], case['observed_sha256']))
    pending = []
    for folder, records, polygon, fill, mask_sha, observed_sha in cases:
        photos = []
        for record in records:
            source = hq / 'CelebA-HQ-img' / (str(record['hq_id']) + '.jpg')
            raw = source.read_bytes()
            if digest(raw) != record['source_sha256']:
                raise ValueError(f'Original photo hash mismatch: {source}')
            with Image.open(io.BytesIO(raw)) as original:
                photo = ImageOps.exif_transpose(original).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
            photos.append(photo)
            pending.append((folder / Path(record['file']).name, photo, record['processed_sha256']))
        mask = Image.new('L', (512, 512), 0)
        ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon], fill=255)
        observed = Image.composite(Image.new('RGB', (512, 512), tuple(fill)), photos[0], mask)
        pending.extend([(folder / 'mask.png', mask, mask_sha), (folder / 'observed.png', observed, observed_sha)])
    # Validate every reconstructed byte and destination before writing anything.
    validated = []
    for destination, photo, expected in pending:
        destination = destination.resolve()
        if not destination.is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError(f'Unsafe output path: {destination}')
        buffer = io.BytesIO()
        photo.save(buffer, format='PNG')
        payload = buffer.getvalue()
        if digest(payload) != expected:
            raise ValueError(f'Restored PNG differs from frozen hash: {destination}. Use the recorded Pillow environment.')
        if destination.exists() and digest(destination.read_bytes()) != expected:
            raise ValueError(f'Existing file differs; refusing to overwrite: {destination}')
        validated.append((destination, payload, expected))
    return validated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hq', type=Path, help='Extracted CelebAMask-HQ root containing CelebA-HQ-img')
    parser.add_argument('--check-only', action='store_true', help='Rebuild in memory and verify without writing photos')
    args = parser.parse_args()
    hq = args.hq
    if hq is None:
        hq = Path(json.loads((ROOT / 'configs/local.json').read_text(encoding='utf-8'))['hq'])
    validated = plan(hq)
    created = 0
    for destination, payload, expected in validated:
        if not args.check_only and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            created += 1
    report = {'mode': 'check_only' if args.check_only else 'restore',
              'cases': 13, 'source_photos': 65, 'verified_pngs': len(validated), 'created': created,
              'files': [{'file': path.relative_to(ROOT).as_posix(), 'sha256': sha} for path, _, sha in validated]}
    (ROOT / 'research/reference_sample_restoration_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(validated)} exact PNGs from 65 original photos across 13 frozen cases; created {created}.')


if __name__ == '__main__':
    main()
