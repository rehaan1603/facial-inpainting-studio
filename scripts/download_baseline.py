"""Download the IOPaint TorchScript export: engineering baseline, not official reproduction."""
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / 'configs/local.json').read_text())
folder = Path(cfg['cache']) / 'models'
folder.mkdir(parents=True, exist_ok=True)
path = folder / 'big-lama.pt'
url = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'
if not path.exists():
    temp = path.with_suffix('.download')
    urllib.request.urlretrieve(url, temp)
    temp.replace(path)
raw = path.read_bytes()
assert hashlib.md5(raw).hexdigest() == 'e3aa4aaa15225a33ec84f9f4bc47e500', 'Published checksum mismatch'
report = {'path': str(path), 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
    'published_md5_verified': True, 'implementation': 'IOPaint/Sanster TorchScript LaMa export',
    'provenance_source': 'https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py',
    'limitations': 'Third-party export. Equivalence to original author checkpoint not independently verified. Not a face-specific trained model.'}
(root / 'research/baseline_provenance.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
