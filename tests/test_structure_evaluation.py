"""Test structural endpoint and gate failures without using any photograph."""
import copy
import json
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_structure_transport import structure_error, METRICS
from report_structure_transport import compute_gate, summary


class StructureEvaluationTests(unittest.TestCase):
    def test_nme_uses_only_masked_target_locations_and_no_alignment(self):
        target = np.tile([20., 20.], (106,1))
        target[53:] = [80., 80.]
        output = target + [3., 4.]
        output[53:] += 50
        kps = np.array([[10.,10.], [20.,10.], [15.,15.], [12.,20.], [18.,20.]])
        mask = np.zeros((100,100), bool); mask[10:30,10:30] = True
        error, count, scale = structure_error(target, output, kps, mask)
        self.assertEqual(count,53)
        self.assertEqual(scale,10)
        self.assertAlmostEqual(error,.5)
        with self.assertRaises(ValueError):
            structure_error(target, output, kps, np.zeros_like(mask))
        half = np.tile([20.5,20.5], (106,1))
        half_mask = np.zeros_like(mask); half_mask[21,21] = True
        self.assertEqual(structure_error(half,half,kps,half_mask)[1],106)

    @staticmethod
    def groups():
        groups = {}
        ok = {'facenet': {'status':'ok'}, 'arcface_conditioning': {'status':'ok'}}
        for arm in ['candidate','single_reference','wrong_identity','scaffold','reference_context','reference','refface','observed']:
            groups[arm] = []
            for identity in ['a','b','c','d']:
                for seed in [17,29]:
                    m = {metric:.1 for metric in METRICS}
                    m.update(facenet_cosine=.82 if arm=='candidate' else .8,
                             structure_nme=.08 if arm=='candidate' else .1,
                             detections=copy.deepcopy(ok), known_pixels_unchanged=True,
                             facenet_gallery_valid_count=3, arcface_conditioning_gallery_valid_count=3)
                    groups[arm].append({'key':f'{identity}_{seed}_{arm}', 'identity':identity, 'seed':seed,
                        'status':'complete', 'evaluation': {'status':'complete','metrics':m,
                        'target_detections':copy.deepcopy(ok), 'gallery_detections':[copy.deepcopy(ok) for _ in range(3)],
                        'structure': {'status':'ok'}}})
        return groups

    def test_good_numerical_screen_still_does_not_authorize_progression(self):
        result = compute_gate(self.groups(), [17,29])
        self.assertTrue(result['passed_numerical_gate'])
        self.assertFalse(result['progression_authorized'])
        json.dumps(result, allow_nan=False)
        json.dumps(summary(self.groups()['candidate'], [17,29]), allow_nan=False)

    def test_missing_identity_cannot_be_dropped_to_pass(self):
        groups = self.groups()
        groups['candidate'][0]['evaluation']['metrics']['facenet_cosine'] = None
        self.assertFalse(compute_gate(groups,[17,29])['passed_numerical_gate'])

    def test_one_good_seed_cannot_hide_ablation_regression_on_other_seed(self):
        groups = self.groups()
        for r in groups['candidate']:
            r['evaluation']['metrics']['facenet_cosine'] = .87 if r['seed']==17 else .83
        for r in groups['wrong_identity']:
            r['evaluation']['metrics']['facenet_cosine'] = .80 if r['seed']==17 else .84
        result = compute_gate(groups,[17,29])
        self.assertFalse(result['passed_numerical_gate'])
        self.assertTrue(any(c.get('seed')==29 and c['control']=='wrong_identity' and not c['passed'] for c in result['checks']))

    def test_identity_gain_does_not_excuse_pixel_regression(self):
        groups = self.groups()
        for r in groups['candidate']:
            r['evaluation']['metrics']['hole_mae'] = .12
        self.assertFalse(compute_gate(groups,[17,29])['passed_numerical_gate'])


if __name__ == '__main__':
    unittest.main()
