"""Fingerprint loaded studio code so the launcher can avoid stale processes."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'webapp/server.py', 'webapp/version.py', 'webapp/finishing.py',
    'webapp/dist/index.html', 'webapp/dist/app.js', 'webapp/dist/style.css',
    'webapp/dist/confidence.html', 'webapp/dist/confidence.js',
    'webapp/dist/confidence.css', 'webapp/dist/framing.js',
    'scripts/inpaint.py', 'scripts/reference_inpaint.py',
    'scripts/studio_reference_inpaint.py', 'scripts/studio_reference_engine.py',
    'scripts/reference_blending.py', 'scripts/studio_match_edges.py',
]


def code_revision(root=ROOT):
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode('utf-8') + b'\0')
        digest.update((Path(root) / name).read_bytes() + b'\0')
    return digest.hexdigest()
