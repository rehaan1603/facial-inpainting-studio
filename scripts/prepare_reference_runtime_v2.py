"""Install and check the isolated reference runtime without changing OS policy.

Run with the project's CUDA-enabled .venv Python. Existing experiment
environments and all downloaded model weights are left intact.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('This tested environment requires Python 3.12.')
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    project_site = ROOT / '.venv/Lib/site-packages'
    if not (project_site / 'torch').is_dir():
        raise SystemExit('Install the project CUDA environment first.')
    environment = cache / 'reference_env_v2'
    python = environment / 'Scripts/python.exe'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
    # Only share the project's base runtime, never inherit the legacy OSOR stack.
    shared = environment / 'Lib/site-packages/shared_project_runtime.pth'
    shared.write_text(project_site.as_posix() + '\n', encoding='utf-8')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r',
                    str(ROOT / 'requirements-reference-v2.txt'), '--report',
                    str(environment / 'installation-report.json')], check=True)
    raw_report = (environment / 'installation-report.json').read_bytes()
    receipt = json.loads(raw_report)
    # Package descriptions can embed unrelated third-party images. Preserve
    # download URLs, archive hashes, dependencies and package versions instead.
    for item in receipt.get('install', []):
        item.get('metadata', {}).pop('description', None)
    receipt['raw_report_sha256'] = hashlib.sha256(raw_report).hexdigest()
    receipt['omitted'] = 'Distribution descriptions; the complete pip report stays in the local environment.'
    (ROOT / 'research/reference-environment-v2-install.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    subprocess.run([str(python), '-c',
                    "import sys; sys.path.insert(0, 'scripts'); "
                    "from reference_inpaint import reconstruct; "
                    "import torch; assert torch.cuda.is_available(); "
                    "print('Reference imports and CUDA are ready')"], cwd=ROOT, check=True)
    frozen = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'research/reference-environment-v2-lock.txt').write_text(frozen, encoding='utf-8')
    print('Reference environment ready. Restart the local studio to use it.')


if __name__ == '__main__':
    main()
