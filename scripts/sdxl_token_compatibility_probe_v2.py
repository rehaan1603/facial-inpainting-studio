"""Versioned backend recovery for the preserved v1 synthetic backward probe."""
import argparse
import json
import math
import sys
import traceback
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import sdxl_token_compatibility_probe_v1 as original


def backend_flags():
    return {name: bool(getattr(torch.backends.cuda, name + '_sdp_enabled')())
            for name in ('flash', 'mem_efficient', 'math', 'cudnn')}


def sanitize(value, path='', invalid=None):
    invalid = [] if invalid is None else invalid
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append(path)
        return None
    if isinstance(value, dict):
        return {key: sanitize(item, path + '.' + key, invalid) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, f'{path}[{index}]', invalid) for index, item in enumerate(value)]
    return value


def finish_receipt(output, exception=None, extra=None):
    try:
        receipt = json.loads(output.read_text(encoding='utf8'))
    except (OSError, ValueError):
        receipt = {'status': 'failed', 'scope': 'Synthetic backward probe; no quality/training evidence',
                   'error': 'Underlying probe did not produce a readable receipt'}
    if receipt.get('status') == 'running':
        receipt.update(status='failed', error='Underlying probe did not finish its receipt; see wrapper exception')
    if exception is not None:
        receipt['wrapper_exception'] = exception
        receipt['status'] = 'failed'
    receipt.update(extra or {})
    invalid = []
    receipt = sanitize(receipt, invalid=invalid)
    if invalid:
        receipt.update(status='failed', invalid_numeric_fields=invalid,
                       invalid_numeric_policy='Nonfinite diagnostics saved as null; never accepted as passing')
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing receipt')
    if not args.protocol.is_file():
        raise ValueError('Missing frozen protocol')
    before = backend_flags()
    failure = None
    inside = None
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            inside = backend_flags()
            original.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            failure = f'Underlying probe exited with code {exc.code}'
    except Exception:
        failure = traceback.format_exc()
    after = backend_flags()
    receipt = finish_receipt(args.output, failure, dict(
        wrapper_sha256=original.sha(__file__), wrapped_script_sha256=original.sha(original.__file__),
        backend_policy=['FLASH_ATTENTION', 'MATH'], backend_flags_before=before,
        backend_flags_inside=inside, backend_flags_after=after, backend_flags_restored=before == after))
    print(json.dumps({key: receipt.get(key) for key in ('status', 'error', 'resolution', 'gradient_norm',
        'neutral_max_absolute_error', 'peak_allocated_bytes', 'backend_flags_restored')}, indent=2), flush=True)
    if receipt['status'] != 'passed' or before != after:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
