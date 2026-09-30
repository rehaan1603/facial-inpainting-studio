"""Frozen, evaluation-only metrics for the structure transport feasibility screen.

No generation is performed here. Clean targets/gallery are accessed only after
all 64 generation/control rows exist. Landmark coordinates remain in memory.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from extended_evaluation_metrics import Metrics, read_rgb

BASE = ROOT / 'outputs/structure_transport_v1'
PROTOCOL = ROOT / 'research/protocols/structure_transport_v1.json'
ORIGIN = ROOT / 'outputs/distortion_aware_v1'
METRICS = ['facenet_cosine', 'facenet_gallery_cosine',
           'arcface_conditioning_cosine', 'arcface_conditioning_gallery_cosine',
           'lpips', 'ssim_rgb', 'psnr_rgb', 'hole_psnr', 'hole_mae',
           'visible_mae', 'niqe', 'brisque', 'structure_nme']
FROZEN_SOURCES = {
    'protocol': PROTOCOL,
    'runner': ROOT / 'scripts/run_structure_transport.py',
    'mechanism': ROOT / 'src/preservation/structure_transport.py',
    'method': ROOT / 'research/STRUCTURE_TRANSPORT_METHOD_V1.md',
    'evaluator': ROOT / 'scripts/evaluate_structure_transport.py',
    'reporter': ROOT / 'scripts/report_structure_transport.py',
    'metric_source': ROOT / 'scripts/extended_evaluation_metrics.py',
    'statistics': ROOT / 'src/preservation/context_statistics.py',
}


def assert_frozen():
    signature = json.loads((BASE / 'signature.json').read_text())
    for name, path in FROZEN_SOURCES.items():
        assert signature[name + '_sha256'] == sha(path), f'Frozen source changed: {name}'
    model_files = signature['model_files']
    for path, digest in model_files.items():
        assert sha(path) == digest, f'Frozen model changed: {Path(path).name}'
    for path, digest in signature['landmark_source_files'].items():
        assert sha(path) == digest, f'Frozen landmark source changed: {Path(path).name}'
    required = {'2d106det.onnx', 'det_10g.onnx'}
    assert required <= {Path(p).name for p in model_files}, 'Missing structural metric model freeze'
    return signature


def validate_rows(data, cfg):
    assert data['all_generation_attempts_finished'] is True
    expected = {(c, s, a) for c in cfg['cases'] for s in cfg['seeds'] for a in cfg['arms']}
    actual = [(r['case_id'], r['seed'], r['mode']) for r in data['rows']]
    assert len(expected) == 64 and len(actual) == 64 and set(actual) == expected
    assert len({r['key'] for r in data['rows']}) == 64
    assert all(r['status'] in ['complete', 'failed'] for r in data['rows']), 'Generation still pending'
    assert all(str(r['identity']) not in {str(i) for i in cfg['reserved_identities']} for r in data['rows'])
    assert data['signature_sha256'] == sha(BASE / 'signature.json')


def structure_error(target_points, output_points, target_kps, mask):
    """Pixel-distance NME, no alignment; select by rounded target pixel only."""
    target_points = np.asarray(target_points, dtype=np.float64)
    output_points = np.asarray(output_points, dtype=np.float64)
    target_kps = np.asarray(target_kps, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if target_points.shape != (106, 2) or output_points.shape != (106, 2) or target_kps.shape != (5, 2):
        raise ValueError('Invalid structural landmark shape')
    if not all(np.isfinite(x).all() for x in [target_points, output_points, target_kps]):
        raise ValueError('Nonfinite structural landmarks')
    interocular = float(np.linalg.norm(target_kps[0] - target_kps[1]))
    if interocular <= 1e-6:
        raise ValueError('Degenerate target interocular distance')
    xy = np.floor(target_points + .5).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    selected = np.zeros(106, dtype=bool)
    selected[inside] = mask[xy[inside, 1], xy[inside, 0]]
    count = int(selected.sum())
    if not count:
        raise ValueError('No target landmarks inside original missing mask')
    value = float(np.linalg.norm(output_points[selected] - target_points[selected], axis=1).mean() / interocular)
    return value, count, interocular


class StructureMetrics:
    """Separate 106-point CPU metric, independent weights from 68-point inference."""
    def __init__(self, model_files):
        import onnxruntime as ort
        from insightface import model_zoo
        from insightface.app.common import Face
        self.Face = Face
        by_name = {Path(p).name: p for p in model_files}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.detector = model_zoo.get_model(by_name['det_10g.onnx'], providers=['CPUExecutionProvider'], sess_options=options)
        self.landmarker = model_zoo.get_model(by_name['2d106det.onnx'], providers=['CPUExecutionProvider'], sess_options=options)
        self.detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=.5)
        self.landmarker.prepare(ctx_id=-1)
        assert self.landmarker.taskname == 'landmark_2d_106'

    def features(self, rgb):
        status = {'status': 'failed', 'face_count': None}
        try:
            bgr = rgb[:, :, ::-1].copy()
            boxes, keypoints = self.detector.detect(bgr, max_num=0, metric='default')
            count = len(boxes)
            status = {'status': 'ok' if count == 1 else 'no_face' if count == 0 else 'multiple_faces', 'face_count': count}
            if count != 1:
                return None, None, status
            face = self.Face(bbox=boxes[0, :4], kps=keypoints[0], det_score=boxes[0, 4])
            self.landmarker.get(bgr, face)
            points = np.asarray(face.landmark_2d_106, dtype=np.float64)
            kps = np.asarray(face.kps, dtype=np.float64)
            if points.shape != (106, 2) or not np.isfinite(points).all():
                raise ValueError('Invalid or nonfinite 106-point output')
            return points, kps, status
        except Exception as error:
            return None, None, dict(status, status='failed', error=f'{type(error).__name__}: {error}')


def main():
    cfg = json.loads(PROTOCOL.read_text())
    signature = assert_frozen()
    data = json.loads((BASE / 'comparison.json').read_text())
    validate_rows(data, cfg)
    assert sha(ORIGIN / 'inference_manifest.json') == signature['manifest_sha256']
    # No target/gallery manifest or image is opened until the complete screen is verified.
    inference = {c['case_id']: c for c in json.loads((ORIGIN / 'inference_manifest.json').read_text())['cases']}
    evaluation = {c['case_id']: c for c in json.loads((ORIGIN / 'evaluation_manifest.json').read_text())['cases']}
    for r in data['rows']:
        assert str(inference[r['case_id']]['identity']) == str(r['identity'])
        assert len(evaluation[r['case_id']]['gallery']) == 3
    evaluation_signature = {
        'comparison_sha256': sha(BASE / 'comparison.json'),
        'generation_signature_sha256': sha(BASE / 'signature.json'),
        'evaluation_manifest_sha256': sha(ORIGIN / 'evaluation_manifest.json'),
        'inference_manifest_sha256': sha(ORIGIN / 'inference_manifest.json'),
        **{k + '_sha256': sha(v) for k, v in FROZEN_SOURCES.items()},
        'model_files': signature['model_files'],
        'structural_metric': '2d106det CPU; one face; rounded target landmark locations in original mask; Euclidean distance / target detector interocular distance; no Procrustes',
        'structural_metric_limit': 'Learned proxy, not anatomical truth; different landmark weights from inference but same vendor/detector family',
        'metric_reuse': 'Within this run only: exact output file hash plus case; geometry and preservation checked each row',
    }
    write_new(BASE / 'evaluation_signature.json', evaluation_signature)
    import cv2
    original_set_threads = cv2.setNumThreads
    cv2.setNumThreads = lambda n: original_set_threads(min(int(n), 1))
    cv2.setNumThreads(1)
    metrics, structure = Metrics(), StructureMetrics(signature['model_files'])
    features, scores, rows = {}, {}, []
    for row in data['rows']:
        score = {'status': 'failed'}
        try:
            if row['status'] != 'complete':
                raise ValueError('Generation failed: ' + row.get('error', 'no result'))
            c, e = inference[row['case_id']], evaluation[row['case_id']]
            output = read_rgb(row['output'], row['output_sha256'])
            observed = read_rgb(c['observed'], c['observed_sha256'])
            mask = read_rgb(c['mask'], c['mask_sha256'])[:, :, 0] >= 128
            if output.shape != observed.shape or mask.shape != output.shape[:2] or not mask.any() or mask.all():
                raise ValueError('Invalid output/observation/mask geometry')
            if not np.array_equal(output[~mask], observed[~mask]):
                raise ValueError('Known pixels changed')
            case_id = row['case_id']
            if case_id not in features:
                target = read_rgb(e['target']['path'], e['target']['sha256'])
                if target.shape != output.shape:
                    raise ValueError('Target/output geometry differs')
                tv, td = metrics.features(target)
                galleries = [metrics.features(read_rgb(g['path'], g['sha256'])) for g in e['gallery']]
                tp, tk, ts = structure.features(target)
                features[case_id] = target, tv, td, galleries, tp, tk, ts
            target, tv, td, galleries, tp, tk, ts = features[case_id]
            if target.shape != output.shape:
                raise ValueError('Target/output geometry differs')
            cache_key = (case_id, row['output_sha256'])
            if cache_key in scores:
                score = copy.deepcopy(scores[cache_key])
                score['reused_same_case_output_metrics'] = True
            else:
                values = metrics.score(output, target, observed, mask, tv)
                values['known_pixels_unchanged'] = True
                ov, _ = metrics.features(output)
                for name in ['facenet', 'arcface_conditioning']:
                    similarities = [float(np.dot(ov[name], v[name])) for v, _ in galleries if v[name] is not None and ov[name] is not None]
                    values[name + '_gallery_cosine'] = float(np.mean(similarities)) if similarities else None
                    values[name + '_gallery_valid_count'] = len(similarities)
                op, _, os = structure.features(output)
                structural = {'target': ts, 'output': os, 'status': 'failed', 'missing_landmark_count': 0}
                values['structure_nme'] = None
                try:
                    if ts['status'] != 'ok' or os['status'] != 'ok':
                        raise ValueError('Structural target/output one-face detection required')
                    value, count, interocular = structure_error(tp, op, tk, mask)
                    values['structure_nme'] = value
                    structural.update(status='ok', missing_landmark_count=count, target_interocular_pixels=interocular)
                except Exception as error:
                    structural['error'] = f'{type(error).__name__}: {error}'
                score = {'status': 'complete', 'metrics': values, 'target_detections': td,
                         'gallery_detections': [s for _, s in galleries], 'structure': structural,
                         'output_sha256': row['output_sha256'], 'reused_same_case_output_metrics': False}
                scores[cache_key] = copy.deepcopy(score)
        except Exception as error:
            score['error'] = f'{type(error).__name__}: {error}'
        write_new(BASE / 'scores' / (row['key'] + '.json'), score)
        rows.append(dict(row, evaluation=score))
        print('STRUCTURE SCORE', len(rows), '/64', row['key'], score['status'], flush=True)
    assert_frozen()
    assert sha(BASE / 'comparison.json') == evaluation_signature['comparison_sha256']
    write_new(BASE / 'evaluation.json', {'signature_sha256': sha(BASE / 'evaluation_signature.json'), 'rows': rows, 'final_test_used': False})


if __name__ == '__main__':
    main()
