"""Run the unstarted SSIM arm after the original supervisor's file-lock failure.

Preserves the original supervisor, partial receipt, and completed epsilon arm.
No training/evaluation settings or pinned sources are changed.
"""
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/masked_sdxl_ssim_continuation_v1'
OUT = BASE / 'recovery_supervision_v1'
LIMIT = 1800


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.25)


def main():
    original = BASE / 'supervision/receipt.json'
    epsilon = BASE / 'epsilon/training'
    receipt = read(epsilon / 'receipt.json')
    config = read(epsilon / 'config.json')
    previous = read(original)
    require(not OUT.exists(), 'Refusing to overwrite an existing recovery attempt')
    require(not (BASE / 'ssim/training').exists(), 'SSIM arm already exists; no automatic retraining')
    require(previous['unrun_arms'] == ['ssim'] and len(previous['arms']) == 1,
            'Original supervisor state is not the expected interrupted state')
    require(receipt['status'] == 'complete' and receipt['completed_updates'] == 32,
            'Control is incomplete')
    require(sha(epsilon / 'receipt.json') == previous['arms'][0]['child_receipt_sha256'],
            'Completed control receipt changed')
    require(sha(epsilon / 'config.json') == receipt['config_sha256'] and
            sha(epsilon / 'trace.json') == receipt['trace_sha256'], 'Control provenance changed')
    for pin in config['source_pins'] + list(receipt['checkpoints'].values()) + [
            config['cache_receipt'], config['probe_receipt'], config['starting_checkpoint']]:
        require(sha(pin['path']) == pin['sha256'], 'Pinned artifact changed: ' + pin['path'])
    worker = ROOT / 'scripts/train_masked_sdxl_ssim_continuation_v1.py'
    cache = Path(read(ROOT / 'configs/local.json')['cache'])
    command = [str(cache / 'reference_env_v2/Scripts/python.exe'), str(worker), '--arm', 'ssim', '--train']
    record = dict(status='starting', reason='Original supervisor WinError 5 on atomic receipt replace after epsilon completed, before SSIM launch',
                  scientific_settings_changed=False, completed_control_retrained=False, arm='ssim',
                  original_supervisor_receipt_sha256=sha(original), control_receipt_sha256=sha(epsilon / 'receipt.json'),
                  recovery_source_sha256=sha(__file__), worker_sha256=sha(worker),
                  protocol_sha256=config['protocol_sha256'], command=command, timeout_seconds=LIMIT)
    partial = original.with_name('receipt.json.partial')
    if partial.is_file():
        record['original_partial_receipt_sha256'] = sha(partial)
    OUT.mkdir(parents=True, exist_ok=False)
    target = OUT / 'receipt.json'
    save(target, record)
    start = time.monotonic()
    process = None
    try:
        with (OUT / 'ssim.log').open('x', encoding='utf8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            record.update(status='running', worker_pid=process.pid)
            save(target, record)
            record['returncode'] = process.wait(timeout=LIMIT)
        child_path = BASE / 'ssim/training/receipt.json'
        child = read(child_path)
        record.update(child_status=child.get('status'), child_completed_updates=child.get('completed_updates'),
                      child_receipt_sha256=sha(child_path))
        require(record['returncode'] == 0 and child.get('status') == 'complete' and
                child.get('completed_updates') == 32, 'SSIM worker did not complete all 32 updates')
        require(sha(epsilon / 'receipt.json') == record['control_receipt_sha256'] and
                sha(original) == record['original_supervisor_receipt_sha256'], 'Preserved artifacts changed')
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if process is not None and process.poll() is None:
            result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            record['termination_returncode'] = result.returncode
    finally:
        record['wall_seconds'] = time.monotonic() - start
        save(target, record)
        print(json.dumps(record, indent=2), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
