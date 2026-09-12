"""Read-only dataset audit; generated manifests never modify source images."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            shape = image.size
            mode = image.mode
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "width": shape[0], "height": shape[1], "mode": mode}
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def main():
    cfg = json.loads((ROOT / 'configs/local.json').read_text())
    hq, lapa = Path(cfg['hq']), Path(cfg['lapa'])
    ids = dict(line.split() for line in (Path(cfg['annotations']) / 'identity_CelebA.txt').read_text().splitlines())
    parts = dict(line.split() for line in Path(cfg['partitions']).read_text().splitlines())
    mapping = [line.split() for line in (hq / 'CelebA-HQ-to-CelebA-mapping.txt').read_text().splitlines()[1:] if line.strip()]
    assert len(mapping) == 30000 and len({r[0] for r in mapping}) == 30000
    names = {'0': 'train', '1': 'val', '2': 'test'}
    rows = []
    for idx, _, original in mapping:
        rows.append({'hq_id': int(idx), 'original_file': original, 'identity': ids[original],
                     'split': names[parts[original]], 'image_path': str(hq / 'CelebA-HQ-img' / f'{idx}.jpg')})
    groups = {s: {r['identity'] for r in rows if r['split'] == s} for s in names.values()}
    overlap = {f'{a}_{b}': len(groups[a] & groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    # Preserve official partition policy if identity-disjoint; fail instead of silently changing it.
    assert not any(overlap.values()), f'Official HQ splits overlap by identity: {overlap}; decide policy explicitly.'
    out = ROOT / 'data/manifests'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'celebahq.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    lapa_rows = []
    for split in ['train', 'val', 'test']:
        for p in sorted((lapa / split / 'images').glob('*')):
            if p.is_file():
                lapa_rows.append({'split': split, 'image_path': str(p),
                    'label_path': str(lapa / split / 'labels' / (p.stem + '.png')),
                    'landmark_path': str(lapa / split / 'landmarks' / (p.stem + '.txt'))})
    assert lapa_rows, 'No LaPa images found'
    missing = [r for r in lapa_rows if not Path(r['label_path']).is_file() or not Path(r['landmark_path']).is_file()]
    with (out / 'lapa.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=lapa_rows[0].keys()); writer.writeheader(); writer.writerows(lapa_rows)
    paths = [Path(r['image_path']) for r in rows + lapa_rows]
    checks, errors, duplicates = [], [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(inspect, paths), 1):
            checks.append(result)
            if 'error' in result: errors.append(result)
            else: duplicates[result['sha256']].append(result['path'])
            if i % 5000 == 0: print(f'Validated {i}/{len(paths)} images', flush=True)
    (out / 'image_integrity.jsonl').write_text('\n'.join(json.dumps(x) for x in checks), encoding='utf-8')
    mask_files = list((hq / 'CelebAMask-HQ-mask-anno').rglob('*.png'))
    covered = {int(p.stem.split('_')[0]) for p in mask_files}
    report = {'hq_images': len(rows), 'hq_splits': dict(Counter(r['split'] for r in rows)),
        'hq_identities_by_split': {k:len(v) for k,v in groups.items()}, 'identity_overlap': overlap,
        'lapa_splits': dict(Counter(r['split'] for r in lapa_rows)), 'lapa_missing_annotations': missing,
        'hq_mask_files': len(mask_files), 'hq_images_with_any_mask': len(covered),
        'image_decode_errors': errors, 'exact_file_duplicate_groups': [v for v in duplicates.values() if len(v)>1],
        'checks': 'All RGB image files fully decoded and SHA256 hashed; masks inventoried only.',
        'limitations': ['No near-duplicate or cross-dataset identity audit yet.',
            'LaPa identity labels not established.', 'Pretrained backbone exposure unknown.',
            'Annotation semantic correctness not established by file integrity.']}
    (ROOT / 'research/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['exact_file_duplicate_groups']}, indent=2), flush=True)
    if errors or missing or len(covered) != 30000: raise RuntimeError('Dataset audit failed; inspect report')


if __name__ == '__main__': main()
