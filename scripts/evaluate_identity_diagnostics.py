"""Independent FaceNet scoring of frozen v3 outputs; no inference or ranking."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    cache = Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])/'identity_eval_v1'
    provenance_path = ROOT/'research/identity_evaluator_provenance_v1.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for name, expected in provenance['files'].items():
        if sha(cache/name) != expected:
            raise ValueError(f'Evaluator hash mismatch: {name}')
    sys.path.insert(0, str(cache))
    sys.path.insert(0, str(cache/'dependencies'))
    from facenet_pytorch import MTCNN, InceptionResnetV1
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    detector = MTCNN(image_size=160, margin=0, min_face_size=20,
                     thresholds=[.6,.7,.7], factor=.709, post_process=True,
                     keep_all=True, device='cpu')
    model = InceptionResnetV1(classify=True, num_classes=8631)
    model.load_state_dict(torch.load(cache/'vggface2.pt', map_location='cpu', weights_only=True))
    model.classify = False
    model.eval()
    manifest_path = ROOT/'outputs/reference_diagnostics_v1/manifest.json'
    metrics_path = ROOT/'outputs/reference_diagnostic_evaluation_v3/metrics.csv'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with metrics_path.open(encoding='utf-8', newline='') as stream:
        original = list(csv.DictReader(stream))
    if len(original) != 96 or len(manifest['cases']) != 12:
        raise ValueError('Unexpected diagnostic population')
    def embed(path, expected):
        if sha(path) != expected:
            raise ValueError(f'Image hash mismatch: {path}')
        with Image.open(path) as source:
            image = source.convert('RGB')
        with torch.inference_mode():
            boxes, probabilities = detector.detect(image)
            count = 0 if boxes is None else len(boxes)
            record = {'face_count': count, 'status': 'ok' if count == 1 else 'no_face' if count == 0 else 'multiple_faces'}
            if count != 1:
                return None, record
            record.update(box=boxes[0].tolist(), detection_probability=float(probabilities[0]))
            crops = detector.extract(image, boxes, save_path=None)
            vector = model(crops)[0].numpy()
            if not np.isfinite(vector).all():
                raise ValueError('Nonfinite embedding')
            return vector, record
    targets = {}
    target_records = {}
    for case in manifest['cases']:
        target = next(x for x in case['images'] if x['role'] == 'target')
        targets[case['case_id']], target_records[case['case_id']] = embed(ROOT/target['file'], target['processed_sha256'])
    rows = []
    for index, old in enumerate(original):
        folder = ROOT/'outputs/reference_diagnostic_evaluation_v3'/f"{old['case_id']}_scale{old['scale']}_strength{old['strength']}"
        path = folder/('result_hard.png' if old['compositor'] == 'hard' else 'result.png')
        vector, detection = embed(path, old['output_sha256'])
        target = targets[old['case_id']]
        score = float(np.dot(vector, target)) if vector is not None and target is not None else None
        rows.append({k: old[k] for k in ['identity','case_id','scale','strength','compositor','output_sha256']} |
                    {'cosine_similarity': score, 'output_detection': detection, 'target_detection': target_records[old['case_id']]})
        if (index+1) % 12 == 0:
            print(f'Scored {index+1}/96 outputs', flush=True)
    groups = {}
    for row in rows:
        key = f"scale{row['scale']}_strength{row['strength']}_{row['compositor']}"
        groups.setdefault(key, []).append(row)
    summaries = {}
    for key, group in groups.items():
        scores = [r['cosine_similarity'] for r in group if r['cosine_similarity'] is not None]
        summaries[key] = {'total':len(group), 'valid':len(scores), 'mean_cosine':float(np.mean(scores)) if scores else None}
    contrasts = []
    for strength in ['1.0','0.99']:
        for compositor in ['hard','poisson']:
            a = {r['identity']:r['cosine_similarity'] for r in groups[f'scale0.8_strength{strength}_{compositor}']}
            b = {r['identity']:r['cosine_similarity'] for r in groups[f'scale0.0_strength{strength}_{compositor}']}
            ids = sorted(i for i in a if a[i] is not None and b[i] is not None)
            delta = np.array([a[i]-b[i] for i in ids])
            rng = np.random.default_rng(20260914)
            boot = delta[rng.integers(0,len(ids),(10000,len(ids)))].mean(1) if ids else None
            contrasts.append({'strength':strength,'compositor':compositor,'joint_valid':len(ids),'identities':ids,
                              'reference_on_minus_off':float(delta.mean()) if ids else None,
                              'ci95':np.quantile(boot,[.025,.975]).tolist() if ids else None})
    result = {'status':'development_only', 'protocol_sha256':sha(ROOT/'research/IDENTITY_DIAGNOSTIC_PROTOCOL_V1.md'),
              'evaluator_source_sha256':sha(Path(__file__)), 'provenance_sha256':sha(provenance_path),
              'manifest_sha256':sha(manifest_path), 'original_metrics_sha256':sha(metrics_path),
              'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
              'summary':summaries,'paired_contrasts':contrasts,'targets':target_records,'rows':rows}
    output = ROOT/'research/identity_diagnostic_results_v1.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'summary':summaries,'paired_contrasts':contrasts},indent=2), flush=True)

if __name__ == '__main__':
    run()
