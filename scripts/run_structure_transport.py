"""Frozen structural intervention on observed development data; no truth/gallery."""
import json
import importlib.util
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.structure_transport import build_target_shape, transport

BASE = ROOT / 'outputs/structure_transport_v1'
PROTOCOL = ROOT / 'research/protocols/structure_transport_v1.json'


def sanitized_observation(observed, missing):
    if observed.dtype != np.uint8 or missing.dtype != np.bool_ or observed.shape != (*missing.shape, 3):
        raise ValueError('Expected matching uint8 RGB and explicit boolean missing mask')
    return np.where(missing[..., None], np.uint8(128), observed)


class LandmarkExtractor:
    def __init__(self, model_dir):
        import cv2
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.app.common import Face
        self.Face = Face
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.app = FaceAnalysis(name=str(model_dir), allowed_modules=['detection', 'landmark_3d_68'],
                                providers=['CPUExecutionProvider'], sess_options=options)
        self.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=.5)

    def shape(self, rgb):
        faces = self.app.get(rgb[:, :, ::-1].copy())
        if len(faces) != 1:
            raise ValueError(f'Geometry requires exactly one detected face; got {len(faces)}')
        shape = np.asarray(faces[0]['landmark_3d_68'])[:, :2].astype(float)
        if shape.shape != (68, 2) or not np.isfinite(shape).all():
            raise ValueError('Nonfinite or incomplete68 landmarks')
        return shape, np.asarray(faces[0].bbox).copy()

    def observed_shape(self, rgb, bbox):
        face = self.Face(bbox=bbox.copy())
        shape = self.app.models['landmark_3d_68'].get(rgb[:, :, ::-1].copy(), face)[:, :2]
        if shape.shape != (68, 2) or not np.isfinite(shape).all():
            raise ValueError('Invalid observed68 landmarks')
        return shape.astype(float)


def main():
    cfg = json.loads(PROTOCOL.read_text())
    manifest = ROOT / 'outputs/distortion_aware_v1/inference_manifest.json'
    context_file = ROOT / cfg['control_sources']['scaffold_reference_reference_context']
    refface_file = ROOT / cfg['control_sources']['refface']
    cases = {c['case_id']: c for c in json.loads(manifest.read_text())['cases']}
    contexts = {(r['case_id'], r['seed'], r['mode']): r for r in json.loads(context_file.read_text())['rows']}
    reffaces = {r['case_id']: r for r in json.loads(refface_file.read_text())['rows'] if r['mode'] == 'reference_0'}
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    model_dir = cache / 'reference_models/insightface/models/buffalo_l'
    allowed = {}
    for cid in cfg['cases']:
        c = cases[cid]
        if str(c['identity']) in {str(i) for i in cfg['reserved_identities']} or len(c['references']) != 4:
            raise ValueError('Reserved identity or invalid reference count in inference manifest')
        for path, digest in [(c['observed'], c['observed_sha256']), (c['mask'], c['mask_sha256'])] + [(r['path'], r['sha256']) for r in c['references']]:
            allowed[str(Path(path).resolve())] = digest
        for seed in cfg['seeds']:
            for mode in ['scaffold', 'reference', 'reference_context']:
                row = contexts[(cid, seed, mode)]
                assert row['status'] == 'complete' and row['identity'] == c['identity']
                allowed[str(Path(row['output']).resolve())] = row['output_sha256']
        row = reffaces[cid]
        assert row['status'] == 'complete' and row['identity'] == c['identity']
        allowed[str(Path(row['output']).resolve())] = row['output_sha256']
    # Whitelist is constructed solely from inference manifests and existing outputs.
    for path, digest in allowed.items():
        assert sha(path) == digest
    def read(path):
        resolved = str(Path(path).resolve())
        if resolved not in allowed:
            raise ValueError('Image path outside frozen inference whitelist')
        assert sha(resolved) == allowed[resolved]
        with Image.open(resolved) as image:
            return np.asarray(image.convert('RGB')).copy()
    BASE.mkdir(exist_ok=False)
    insight_source = Path(importlib.util.find_spec('insightface').origin).parent
    landmark_source_files = ['model_zoo/landmark.py', 'model_zoo/scrfd.py', 'model_zoo/model_zoo.py',
                             'app/face_analysis.py', 'utils/face_align.py', 'utils/transform.py']
    signature = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(PROTOCOL),
        'runner_sha256': sha(__file__), 'mechanism_sha256': sha(ROOT/'src/preservation/structure_transport.py'),
        'method_sha256': sha(ROOT/'research/STRUCTURE_TRANSPORT_METHOD_V1.md'),
        'evaluator_sha256': sha(ROOT/'scripts/evaluate_structure_transport.py'),
        'reporter_sha256': sha(ROOT/'scripts/report_structure_transport.py'),
        'metric_source_sha256': sha(ROOT/'scripts/extended_evaluation_metrics.py'),
        'statistics_sha256': sha(ROOT/'src/preservation/context_statistics.py'),
        'manifest_sha256': sha(manifest), 'control_comparison_sha256': sha(context_file),
        'refface_comparison_sha256': sha(refface_file),
        'model_files': {str(model_dir/name): sha(model_dir/name) for name in cfg['model_files']},
        'landmark_source_files': {str(insight_source/name): sha(insight_source/name) for name in landmark_source_files},
        'inference_image_count': len(allowed), 'inference_scope': cfg['inference_scope'],
        'final_test_used': False}
    write_new(BASE/'signature.json', signature)
    write_new(BASE/'input_hashes.json', allowed)
    extractor = LandmarkExtractor(model_dir)
    reference_shapes, reference_errors = {}, {}
    for cid in cfg['cases']:
        for ref_index, reference in enumerate(cases[cid]['references']):
            try:
                reference_shapes[(cid, ref_index)] = extractor.shape(read(reference['path']))[0]
            except Exception as error:
                reference_errors[(cid, ref_index)] = f'{type(error).__name__}: {error}'
    rows, zero_checks = [], []
    for case_index, cid in enumerate(cfg['cases']):
        c = cases[cid]
        observed = read(c['observed'])
        missing = read(c['mask'])[:, :, 0] >= 128
        donor = cfg['cases'][(case_index+1) % len(cfg['cases'])]
        for seed in cfg['seeds']:
            scaffold_row = contexts[(cid, seed, 'scaffold')]
            scaffold = read(scaffold_row['output'])
            assert np.array_equal(scaffold[~missing], observed[~missing])
            preparation_error = None
            try:
                source_shape, bbox = extractor.shape(scaffold)
                observed_shape = extractor.observed_shape(sanitized_observation(observed, missing), bbox)
                # No new reconstruction; this checks intervention-off pixel equivalence.
                interior = missing[:-1, :-1] & missing[1:, :-1] & missing[:-1, 1:] & missing[1:, 1:]
                control_y, control_x = np.nonzero(interior)
                if not len(control_x):
                    raise ValueError('No mask interior for zero-displacement integrity check')
                control = np.array([[control_x[0], control_y[0]]], dtype=float)
                same, meta = transport(scaffold, observed, missing, control, control,
                    regularization=cfg['parameters']['regularization'], max_displacement_fraction=cfg['parameters']['maximum_displacement_fraction'])
                assert np.array_equal(same, scaffold)
                zero_checks.append({'case_id': cid, 'seed': seed, 'bit_exact': True})
            except Exception as error:
                preparation_error = f'{type(error).__name__}: {error}'
                zero_checks.append({'case_id': cid, 'seed': seed, 'bit_exact': False, 'error': preparation_error})
            for mode in cfg['arms']:
                key = f'{cid}_s{seed}_{mode}'
                row = {'key': key, 'case_id': cid, 'identity': c['identity'], 'seed': seed, 'mode': mode, 'status': 'failed'}
                if mode in ['scaffold', 'reference_context', 'reference']:
                    prior = contexts[(cid, seed, mode)]
                    row.update(status='complete', output=prior['output'], output_sha256=prior['output_sha256'], reused_control=True)
                elif mode in ['refface', 'observed']:
                    prior = reffaces[cid] if mode == 'refface' else {'output': c['observed'], 'output_sha256': c['observed_sha256']}
                    row.update(status='complete', output=prior['output'], output_sha256=prior['output_sha256'], reused_control=True,
                               deterministic_shared_across_seed_slots=True)
                else:
                    start = time.monotonic()
                    try:
                        if preparation_error:
                            raise ValueError(preparation_error)
                        reference_case = donor if mode == 'wrong_identity' else cid
                        ref_indices = [0] if mode == 'single_reference' else cfg['reference_indices']
                        refs = []
                        for ref_index in ref_indices:
                            ref_key = (reference_case, ref_index)
                            if ref_key in reference_errors:
                                raise ValueError(f'Reference {reference_case}/{ref_index}: {reference_errors[ref_key]}')
                            refs.append(reference_shapes[ref_key])
                        desired, alignment = build_target_shape(observed_shape, refs, missing,
                            anchor_margin=cfg['parameters']['anchor_margin'], minanchors=cfg['parameters']['minimum_anchors'])
                        output, meta = transport(scaffold, observed, missing, source_shape, desired,
                            regularization=cfg['parameters']['regularization'], max_displacement_fraction=cfg['parameters']['maximum_displacement_fraction'])
                        assert np.array_equal(output[~missing], observed[~missing])
                        folder = BASE/'generations'; folder.mkdir(exist_ok=True)
                        path = folder/(key+'.png'); Image.fromarray(output).save(path)
                        geom_folder = BASE/'geometry'; geom_folder.mkdir(exist_ok=True)
                        geom = geom_folder/(key+'.npz')
                        np.savez_compressed(geom, source=source_shape, observed=observed_shape, proposed=desired, references=np.stack(refs))
                        row.update(status='complete', output=str(path), output_sha256=sha(path), geometry_sha256=sha(geom),
                            known_pixels_unchanged=True, reference_count=len(refs), reference_case=reference_case,
                            alignment=alignment, deformation=meta, generation_seconds=time.monotonic()-start)
                    except Exception as error:
                        row.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
                        if hasattr(error, 'metadata'):
                            row['failure_metadata'] = error.metadata
                    row['attempt_seconds'] = time.monotonic()-start
                rows.append(row)
                write_new(BASE/'receipts'/(key+'.json'), row)
                print('STRUCTURE', len(rows), '/64', key, row['status'], row.get('error',''), flush=True)
    write_new(BASE/'zero_equivalence.json', {'rows': zero_checks, 'image_transforms_only': True})
    write_new(BASE/'comparison.json', {'rows': rows, 'signature_sha256': sha(BASE/'signature.json'),
        'all_generation_attempts_finished': True, 'final_test_used': False, 'new_neural_generations': 0})


if __name__ == '__main__':
    main()
