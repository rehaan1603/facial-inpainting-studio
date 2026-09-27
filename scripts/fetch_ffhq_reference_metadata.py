"""Read only the six official FFHQ-Ref split/mapping entries via HTTP ranges.

No archive image entry is requested or decoded. This inspects data availability;
it does not select experimental people or certify author-predicted identities.
"""
import hashlib
import json
import struct
import sys
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new

URL = 'https://github.com/ChiWeiHsiao/ref-ldm/releases/download/0.1.0/FFHQ-Ref.zip'
SIZE = 446546859  # GitHub release asset length, independently checked by Content-Range
DEST = ROOT / 'outputs/ffhq_ref_metadata_v1'
ALLOWED = {f'reference_mapping/{s}_references.csv' for s in ['train', 'val', 'test']} | {
    f'id_based_ffhq_split/{s}_image.txt' for s in ['train', 'val', 'test']}


def fetch(start, end):
    request = urllib.request.Request(URL, headers={
        'Range': f'bytes={start}-{end}', 'User-Agent': 'FacialInpaintingResearch/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        assert response.status == 206, 'Server ignored partial request; refusing archive download'
        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}'
        content = response.read(end - start + 2)
        assert len(content) == end - start + 1
        return content


def main():
    assert not DEST.exists(), 'Metadata receipt already exists; preserve it'
    tail = fetch(SIZE-65536, SIZE-1)
    offset = tail.rfind(b'PK\x05\x06')
    assert offset >= 0
    _, disk, central_disk, n_disk, count, length, start, comment = struct.unpack_from('<4s4H2IH', tail, offset)
    assert disk == central_disk == 0 and n_disk == count and comment == 0
    central = fetch(start, start + length - 1)
    position, selected, seen = 0, {}, 0
    while position < len(central):
        entry = struct.unpack_from('<4s6H3I5H2I', central, position)
        assert entry[0] == b'PK\x01\x02'
        _, _, _, flags, method, _, _, crc, compressed, uncompressed, fn, extra, note, _, _, _, local = entry
        filename = central[position+46:position+46+fn].decode('utf-8')
        # Strictly whitelist the documented root metadata files, not __MACOSX copies.
        relative = filename.removeprefix('FFHQ-Ref/')
        if filename.startswith('FFHQ-Ref/') and relative in ALLOWED:
            assert relative not in selected
            selected[relative] = dict(crc=crc, compressed=compressed, uncompressed=uncompressed,
                                      method=method, local=local, flags=flags)
        position += 46 + fn + extra + note
        seen += 1
    assert seen == count and set(selected) == ALLOWED, list(selected)
    records = []
    for name, info in sorted(selected.items()):
        header = fetch(info['local'], info['local']+29)
        values = struct.unpack('<4s5H3I2H', header)
        assert values[0] == b'PK\x03\x04' and not (info['flags'] & 1)
        payload_start = info['local']+30+values[-2]+values[-1]
        data = fetch(payload_start, payload_start+info['compressed']-1)
        content = zlib.decompress(data, -15) if info['method'] == 8 else data
        assert info['method'] in [0, 8] and len(content) == info['uncompressed']
        assert zlib.crc32(content) == info['crc']
        content.decode('utf-8')  # All six entries must be text.
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
        records.append({'name': name, 'sha256': hashlib.sha256(content).hexdigest(),
                        'bytes': len(content), 'compressed_bytes': info['compressed']})
    receipt = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'source_url': URL,
               'archive_bytes': SIZE, 'archive_entry_count': count,
               'central_directory_sha256': hashlib.sha256(central).hexdigest(),
               'files': records, 'image_entries_requested': 0, 'image_pixels_opened': False,
               'script_sha256': sha(__file__),
               'scope': 'Author-predicted grouping metadata only; no image selection or evaluation'}
    write_new(DEST / 'receipt.json', receipt)
    write_new(ROOT / 'research/ffhq_ref_metadata_receipt_v1.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
