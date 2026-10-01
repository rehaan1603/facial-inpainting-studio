"""Retain paired support and failures; exploratory pilot never establishes significance."""
import json
import numpy as np
from reference_intervention_core import ROOT, sha


def main():
    out = ROOT/'outputs/reference_intervention_pilot_v1'
    scores = json.loads((out/'scores.json').read_text())
    indexed = {(r['identity'], r['arm'], r['condition']):r for r in scores['rows']}
    ids = sorted({r['identity'] for r in scores['rows']})
    comparisons = []
    for condition in ['clean','corrupt']:
        for control in ['baseline','reconstruction']:
            for metric in ['facenet_cosine','arcface_conditioning_cosine','lpips','mae']:
                values, supported = [], []
                for identity in ids:
                    a = indexed[identity,'intervention',condition]['metrics'].get(metric)
                    b = indexed[identity,control,condition]['metrics'].get(metric)
                    if a is not None and b is not None:
                        values.append(a-b); supported.append(identity)
                comparisons.append(dict(condition=condition, contrast='intervention - '+control, metric=metric,
                    common_support=supported, n=len(values), identity_deltas=values,
                    mean_delta=float(np.mean(values)) if values else None))
    failures = []
    for r in scores['rows']:
        for family in ['target_detections','output_detections']:
            for metric, entry in r[family].items():
                if entry['status'] != 'ok':
                    failures.append(dict(identity=r['identity'], arm=r['arm'], condition=r['condition'],
                                         family=family, metric=metric, status=entry['status']))
    result = dict(score_sha256=sha(out/'scores.json'), analysis_script_sha256=sha(__file__),
        comparisons=comparisons, detection_events=failures,
        decision='Do not promote: corrupted-reference FaceNet regresses against baseline. Novelty is not established.',
        inference='Descriptive pilot only. Four identities, one generation seed, no significance claim.',
        review=dict(sheets_reviewed=ids, blinded=False,
                    observation='Outputs are sharper than degraded inputs, but target-specific gaze, skin detail and mouth shape are not exact. No clear consistent visual advantage of intervention over reconstruction control.'),
        milestones=dict(literature_overlap_check='partial; generic reliability overlaps RefSTAR',
            adapter_implementation='complete for this prototype', unit_tests='4 passed',
            pilot_data_and_encoding='complete: 16 train and 4 adapter-heldout identities',
            paired_training='complete: 64 steps per arm',
            generation='complete: 28 images', metric_evaluation='complete: 28 rows with missing identity scores retained',
            independent_benchmark='not run', novelty='not established', production_promotion='rejected'))
    (ROOT/'research/reference_intervention_pilot_analysis_v1.json').write_text(json.dumps(result,indent=2))
    public_training = json.loads((out/'training/receipt.json').read_text())
    public_training.update(cache_receipt_sha256=sha(out/'cache_receipt.json'), manifest_sha256=sha(out/'manifest.json'),
        train_identities=[c['identity'] for c in json.loads((out/'manifest.json').read_text())['cases'] if c['role']=='train'],
        adapter_holdout_identities=ids, final_test_used=False)
    (ROOT/'research/reference_intervention_pilot_training_v1.json').write_text(json.dumps(public_training,indent=2))
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    main()
