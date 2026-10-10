"""Bound one isolated worker; keep every failed/interrupted receipt and studio."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, choices=(300,), default=300)
    args = parser.parse_args()
    supervisor = args.output.with_name(args.output.stem + '_supervisor.json')
    if args.output.exists() or supervisor.exists():
        raise ValueError('Refusing to overwrite a previous attempt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    command = [str(cache/'reference_env_v2/Scripts/python.exe'),
               str(ROOT/'scripts/sdxl_token_compatibility_probe_v2.py'),
               '--resolution', str(args.resolution), '--protocol', str(args.protocol.resolve()),
               '--output', str(args.output.resolve())]
    receipt = dict(status='running', scope='Bounded synthetic gradient/memory attempt, no quality claim',
                   timeout_seconds=args.timeout, resolution=args.resolution, command=command,
                   protocol_sha256=sha(args.protocol), supervisor_sha256=sha(__file__),
                   worker_sha256=sha(ROOT/'scripts/sdxl_token_compatibility_probe_v2.py'),
                   core_sha256=sha(ROOT/'scripts/sdxl_token_compatibility_probe_v1.py'))
    with supervisor.open('x', encoding='utf8') as handle:
        json.dump(receipt, handle, indent=2)
    start = time.monotonic()
    log = args.output.with_suffix('.log')
    with log.open('x', encoding='utf8') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['worker_pid'] = process.pid
        save(supervisor, receipt)
        try:
            returncode = process.wait(timeout=args.timeout)
            receipt.update(status='finished', returncode=returncode, timeout=False)
        except subprocess.TimeoutExpired:
            # Popen's PID belongs to this exact worker launch. Windows venv
            # launchers can spawn a real Python child, so terminate that own tree.
            killed = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            receipt.update(status='failed', timeout=True, error='Worker exceeded fixed wall-clock budget',
                           termination_returncode=killed.returncode)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                receipt['termination_incomplete'] = True
            if args.output.exists():
                child = json.loads(args.output.read_text(encoding='utf8'))
            else:
                child = dict(scope='No completed worker receipt')
            child.update(status='failed', error='Fixed 300-second wall-clock budget exceeded; no passing gradient/update inferred',
                         timed_out=True, exact_stuck_stage='unknown', checkpoint_saved=False)
            save(args.output, child)
    receipt['wall_seconds'] = time.monotonic() - start
    save(supervisor, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    child = json.loads(args.output.read_text(encoding='utf8')) if args.output.exists() else {}
    if receipt.get('timeout') or receipt.get('returncode') != 0 or child.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
