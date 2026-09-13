"""Record local output image paths and hashes without embedding image content."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for path in sorted((ROOT / 'outputs').rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            records.append({'file': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': checksum})
    report = {'scope': 'Local outputs snapshot; includes inputs, references, masks, results and previews. This is an inventory, not a license grant or quality endorsement.',
              'image_count': len(records), 'total_bytes': sum(item['bytes'] for item in records), 'files': records}
    destination = ROOT / 'research/LOCAL_IMAGE_INVENTORY.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Indexed {len(records)} local output images; photo contents remain local.')


if __name__ == '__main__':
    main()
