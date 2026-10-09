"""Optional boundary matching for single-image studio results."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def match_edges(folder, mode):
    folder = Path(folder)
    if mode not in {'hard', 'poisson'}:
        raise ValueError('Unsupported edge matching mode.')
    raw, result = folder / 'result_raw.png', folder / 'result.png'
    if mode == 'hard':
        shutil.copyfile(raw, result)
        return {'requested_mode': mode, 'mode': 'hard', 'fallback': False}, []
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    python = cache / 'reference_env_v2/Scripts/python.exe'
    if not python.is_file():
        shutil.copyfile(raw, result)
        return {'requested_mode': mode, 'mode': 'hard', 'fallback': True,
                'fallback_reason': 'edge_matching_environment_missing'}, [
                    'Edge matching is unavailable; showing the raw reconstruction.']
    command = [str(python), str(ROOT / 'scripts/studio_match_edges.py'),
               '--input', str(folder / 'input.png'), '--prediction', str(raw),
               '--mask', str(folder / 'effective_mask.png'), '--output', str(result),
               '--metadata', str(folder / 'blending.json')]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        with (folder / 'edge_matching.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True,
                           timeout=60, creationflags=flags)
        metadata = json.loads((folder / 'blending.json').read_text())
        if not isinstance(metadata, dict) or not result.is_file():
            raise ValueError('The finishing worker did not return a valid result.')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        shutil.copyfile(raw, result)
        return {'requested_mode': mode, 'mode': 'hard', 'fallback': True,
                'fallback_reason': 'edge_matching_worker_failed',
                'failure_type': type(error).__name__}, [
                    'Edge matching could not finish; showing the raw reconstruction.']
    warnings = []
    if metadata.get('fallback'):
        warnings.append('Edge matching could not use this mask; showing the raw reconstruction.')
    return metadata, warnings
