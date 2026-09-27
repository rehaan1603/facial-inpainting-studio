"""Download publicly linked author checkpoints to a private local cache.

This records provenance and hashes; it never deserializes checkpoint contents.
No dataset or photographic asset is downloaded by this script.
"""
import hashlib
from html.parser import HTMLParser
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new

ASSETS = {
    'model.pth': '1qn1fKj-4iwykSZl_GT9kjz2UTnbMlU36',
    'BEST_checkpoint_r101.pth': '1VpD27jHOPaOJRKFqOO_txLHAE05CbtLU',
}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}; self.inside = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'form' and values.get('id') == 'download-form':
            self.action = values['action']; self.inside = True
        if tag == 'input' and self.inside and values.get('type') == 'hidden':
            self.fields[values['name']] = values['value']

    def handle_endtag(self, tag):
        if tag == 'form': self.inside = False


def main():
    cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])/'refface_weights_v1'
    cache.mkdir(exist_ok=True)
    all_records = []
    for name, drive_id in ASSETS.items():
        record_path = cache/(name+'.receipt.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['status'] == 'complete' and sha(cache/name) == record['sha256']
            all_records.append(record); continue
        partial = cache/(name+'.partial')
        assert not partial.exists() and not (cache/name).exists(), 'Unreceipted asset: inspect before retrying'
        url = f'https://drive.google.com/uc?export=download&id={drive_id}'
        response = urllib.request.urlopen(url, timeout=45)
        if 'text/html' in response.headers.get('Content-Type', ''):
            page = response.read(200000).decode('utf-8'); response.close()
            form = DownloadForm(); form.feed(page)
            assert form.action and urllib.parse.urlparse(form.action).netloc == 'drive.usercontent.google.com', 'Public download not available'
            assert form.fields.get('id') == drive_id
            response = urllib.request.urlopen(form.action+'?'+urllib.parse.urlencode(form.fields), timeout=60)
        with response:
            assert response.status == 200 and 'text/html' not in response.headers.get('Content-Type', ''), 'Expected checkpoint response'
            expected = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
            if expected is not None: assert 0 < expected < 2_500_000_000
            total, digest = 0, hashlib.sha256()
            with partial.open('xb') as stream:
                while True:
                    block = response.read(4*1024*1024)
                    if not block: break
                    total += len(block)
                    assert total <= 2_500_000_000
                    stream.write(block); digest.update(block)
            assert total and (expected is None or total == expected)
        partial.rename(cache/name)
        record = {'name': name, 'source_url': f'https://drive.google.com/file/d/{drive_id}/view',
                  'source_listing': 'https://github.com/WuyangLuo/RefFaceInpainting/blob/0f1ad75677cc8fae4ae14d878e4c6cfce9365f28/README.md',
                  'bytes': total, 'sha256': digest.hexdigest(), 'status': 'complete',
                  'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'author_published_digest_available': False, 'deserialized': False}
        write_new(record_path, record); all_records.append(record)
        print(name, total, record['sha256'], flush=True)
    write_new(ROOT/'research/refface_downloads_v1.json', {
        'files': all_records, 'script_sha256': sha(__file__),
        'scope': 'Public author checkpoint acquisition, not model reproduction or an accuracy result'})


if __name__ == '__main__': main()
