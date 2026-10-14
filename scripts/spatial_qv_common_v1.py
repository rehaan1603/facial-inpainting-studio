"""Immutable-input and reliable receipt helpers for the isolated Q/V experiment."""
import hashlib
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'research/MASKED_SPATIAL_QV_PREFLIGHT_V1.md'
CONTROL = ROOT / 'outputs/masked_sdxl_ssim_continuation_v1/ssim/training'
START_SHA = '0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba'
CONTROL_SHA = '9d4aee8cb4a537e81a78a01c2ed20d0fb8971402687c83b3c47fde518a89cbe3'
IDS = ['6792', '2250', '2083', '4164', '674', '476', '1281', '8748']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verify(record):
    require(sha(record['path']) == record['sha256'], 'Pinned artifact changed: ' + record['path'])


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf8'))
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.25)


def verify_control():
    cfg, receipt, trace = [read(CONTROL / name) for name in ['config.json', 'receipt.json', 'trace.json']]
    require(receipt['status'] == 'complete' and receipt['completed_updates'] == receipt['finite_gradient_updates'] == 32,
            'Same-loss control incomplete')
    require(cfg['arm'] == 'ssim' and cfg['ssim_weight'] == .1 and cfg['training_ids'] == IDS and cfg['updates'] == 32,
            'Same-loss control recipe changed')
    require(cfg['starting_checkpoint']['sha256'] == START_SHA and receipt['checkpoints']['last']['sha256'] == CONTROL_SHA,
            'Control initialization or final checkpoint changed')
    require(sha(CONTROL / 'config.json') == receipt['config_sha256'] and sha(CONTROL / 'trace.json') == receipt['trace_sha256'],
            'Control trace/config pins changed')
    require(receipt['backend_flags_restored'] and not receipt['frozen_weights_have_gradients'] and
            not receipt['frozen_vae_have_gradients'], 'Control frozen-weight/backend guard failed')
    require(len(trace) == 32 and all(row['update'] == i+1 and row['identity'] == IDS[i % 8] and
            row['timestep'] == [100, 300][i // 16] and row['noise_seed'] == 20261011+i
            for i, row in enumerate(trace)), 'Control schedule changed')
    for item in cfg['source_pins'] + list(receipt['checkpoints'].values()) + [
            cfg['starting_checkpoint'], cfg['cache_receipt'], cfg['probe_receipt'], cfg['prior_training_receipt']]:
        verify(item)
    return cfg, receipt, trace


def verify_cpu_tests():
    path = ROOT / 'outputs/masked_spatial_qv_cpu_tests_v1.json'
    evidence = read(path)
    require(evidence['status'] == 'passed' and evidence['failures'] == 0 and evidence['tests'] > 0,
            'Passing CPU tests required')
    for item in evidence['source_pins']:
        verify(item)
    return pin(path)
