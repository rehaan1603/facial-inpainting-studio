"""Cache pinned official FaceNet sources and weights without changing environments."""
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REV = '787da06156087cd6b616fe6608213722bddc30cd'

def prepare():
    cache = Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])
    dest = cache/'identity_eval_v1'
    dest.mkdir(parents=True, exist_ok=True)
    dependencies = dest/'dependencies'
    if not (dependencies/'requests').exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(dependencies),
                        '-r', str(ROOT/'requirements-identity-eval.txt')], check=True)
    source = f'https://codeload.github.com/timesler/facenet-pytorch/zip/{REV}'
    package = dest/'facenet_pytorch'
    if not package.exists():
        raw = urllib.request.urlopen(source, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            prefix = f'facenet-pytorch-{REV}/'
            for item in archive.infolist():
                relative = item.filename.removeprefix(prefix)
                if not relative or item.is_dir():
                    continue
                path = (package/relative).resolve()
                if not path.is_relative_to(package.resolve()):
                    raise ValueError('Unsafe archive path')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))
    url = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    weight = dest/'vggface2.pt'
    if not weight.exists():
        temporary = weight.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(weight)
    files = [weight, *sorted(package.rglob('*.py')), *sorted(package.rglob('*.pt')),
             *sorted(dependencies.rglob('*.py')), *sorted(dependencies.rglob('*.pyd')),
             *sorted(dependencies.rglob('METADATA')), package/'LICENSE.md']
    record = {'commit': REV, 'source_url': source, 'weight_url': url,
              'files': {str(p.relative_to(dest)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}}
    (ROOT/'research/identity_evaluator_provenance_v1.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Pinned evaluator cached:', dest, flush=True)

if __name__ == '__main__':
    prepare()
