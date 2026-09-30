import unittest
from unittest.mock import patch

import numpy as np

from src.preservation.structure_transport import (
    StructureTransportError, build_target_shape, similarity_fit, transform_points, transport,
)


class StructureTransportTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((64, 64), bool)
        self.mask[16:48, 16:48] = True
        yy, xx = np.indices(self.mask.shape)
        self.scaffold = np.stack([3 * xx, 3 * yy, xx + yy], axis=2).astype(np.uint8)
        self.observed = np.full(self.scaffold.shape, 17, np.uint8)

    def test_similarity_recovers_rotation_scale_translation_without_mutation(self):
        source = np.array([[0., 0.], [4., 0.], [0., 3.], [2., 5.]])
        original = source.copy()
        angle = .3
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        expected = source @ (1.7 * rotation).T + [8, -5]
        fit = similarity_fit(source, expected)
        np.testing.assert_allclose(transform_points(source, fit), expected, atol=1e-12)
        self.assertGreater(np.linalg.det(fit[:, :2]), 0)
        np.testing.assert_array_equal(source, original)

    def test_degenerate_and_nonfinite_fits_rejected(self):
        for points in [np.zeros((6, 2)), np.column_stack([np.arange(6), np.arange(6)]),
                       np.full((6, 2), np.nan)]:
            with self.subTest(points=points):
                with self.assertRaises(StructureTransportError):
                    similarity_fit(points, points)

    def _shapes(self):
        # Visible anchors surround a central hole; the other 60 coordinates are
        # deliberately missing and must not affect reference registration.
        anchors = np.array([[12, 12], [32, 12], [52, 12], [12, 32],
                            [52, 32], [12, 52], [32, 52], [52, 52]], float)
        shape = np.vstack([anchors, np.tile([32., 32.], (60, 1))])
        mask = np.zeros((64, 64), bool)
        mask[25:40, 25:40] = True
        return shape, mask

    def test_consensus_uses_only_safe_anchors_and_actual_reference_geometry(self):
        observed, mask = self._shapes()
        references = []
        for offset in [-2, 0, 2, 8]:
            shape = observed.copy()
            shape[8:, 0] += offset
            references.append(shape)
        desired, meta = build_target_shape(observed, references, mask)
        self.assertEqual(meta['anchor_indices'], list(range(8)))
        np.testing.assert_allclose(desired[8:, 0], 33.)
        # Alter only inferred missing target coordinates, while keeping them in
        # the hole. They cannot change transforms or the resulting consensus.
        other = observed.copy()
        other[8:] = [28, 29]
        second, _ = build_target_shape(other, references, mask)
        np.testing.assert_allclose(second, desired)

    def test_missing_or_collinear_anchors_rejected(self):
        observed, mask = self._shapes()
        with self.assertRaisesRegex(StructureTransportError, 'insufficient'):
            build_target_shape(observed, [observed], np.ones_like(mask))
        observed[:8] = np.column_stack([np.arange(8) + 15, np.repeat(12., 8)])
        with self.assertRaisesRegex(StructureTransportError, 'degenerate'):
            build_target_shape(observed, [observed], mask)

    def test_zero_displacement_is_byte_exact_composition(self):
        points = np.array([[30., 30.], [36., 36.]])
        result, meta = transport(self.scaffold, self.observed, self.mask, points, points)
        expected = self.scaffold.copy()
        expected[~self.mask] = self.observed[~self.mask]
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(meta['zero_displacement'])
        self.assertEqual(meta['minimum_changed_inverse_jacobian'], 1)

    def test_inverse_direction_preservation_and_input_immutability(self):
        source, desired = np.array([[30., 32.]]), np.array([[32., 32.]])
        originals = [a.copy() for a in [self.scaffold, self.observed, self.mask, source, desired]]
        result, meta = transport(self.scaffold, self.observed, self.mask, source, desired)
        # The desired coordinate must sample to its left (lower red channel),
        # not to its right. Regularization makes the constraint intentionally soft.
        self.assertLess(int(result[32, 32, 0]), int(self.scaffold[32, 32, 0]))
        self.assertGreater(int(result[32, 32, 0]), int(self.scaffold[32, 29, 0]))
        np.testing.assert_array_equal(result[~self.mask], self.observed[~self.mask])
        for value, old in zip([self.scaffold, self.observed, self.mask, source, desired], originals):
            np.testing.assert_array_equal(value, old)
        self.assertGreater(meta['minimum_changed_inverse_jacobian'], 0)

    def test_missing_observation_payload_never_reaches_output(self):
        source, desired = np.array([[30., 32.]]), np.array([[32., 32.]])
        altered = self.observed.copy()
        altered[self.mask] = 251
        a, _ = transport(self.scaffold, self.observed, self.mask, source, desired)
        b, _ = transport(self.scaffold, altered, self.mask, source, desired)
        np.testing.assert_array_equal(a, b)

    def test_disconnected_constraints_rejected_and_unconstrained_island_unchanged(self):
        mask = np.zeros_like(self.mask)
        mask[10:25, 10:25] = True
        mask[36:51, 36:51] = True
        with self.assertRaisesRegex(StructureTransportError, 'no admissible'):
            transport(self.scaffold, self.observed, mask, [[16., 16.]], [[42., 42.]])
        result, meta = transport(self.scaffold, self.observed, mask,
                                 [[16., 16.], [16., 16.]], [[17., 16.], [42., 42.]])
        self.assertEqual(meta['constraint_count'], 1)
        self.assertEqual(meta['rejected_constraints']['outside_or_different_mask_component'], 1)
        np.testing.assert_array_equal(result[36:51, 36:51], self.scaffold[36:51, 36:51])

    def test_all_four_bilinear_pixels_must_be_masked_even_at_integer_constraint(self):
        with self.assertRaisesRegex(StructureTransportError, 'no admissible'):
            transport(self.scaffold, self.observed, self.mask, [[46., 30.]], [[47., 30.]])

    def test_fractional_bilinear_constraint_moves_in_expected_direction(self):
        result, meta = transport(self.scaffold, self.observed, self.mask,
                                 [[30.25, 30.75]], [[31.25, 31.75]])
        self.assertEqual(meta['constraint_count'], 1)
        self.assertLess(int(result[32, 31, 0]), int(self.scaffold[32, 31, 0]))
        self.assertLess(int(result[32, 31, 1]), int(self.scaffold[32, 31, 1]))

    def test_displacement_bound_and_nonfinite_solver_rejected(self):
        with self.assertRaisesRegex(StructureTransportError, 'exceeds frozen bound'):
            transport(self.scaffold, self.observed, self.mask,
                      [[30., 32.]], [[32., 32.]], max_displacement_fraction=.001)
        with patch('src.preservation.structure_transport.spsolve',
                   return_value=np.full((int(self.mask.sum()), 2), np.nan)):
            with self.assertRaisesRegex(StructureTransportError, 'nonfinite'):
                transport(self.scaffold, self.observed, self.mask, [[30., 32.]], [[32., 32.]])

    def test_folding_solver_field_rejected(self):
        ys, xs = np.nonzero(self.mask)
        displacement = np.column_stack([-2. * (xs - 32), np.zeros(len(xs))])
        with patch('src.preservation.structure_transport.spsolve', return_value=displacement):
            with self.assertRaisesRegex(StructureTransportError, 'Jacobian'):
                transport(self.scaffold, self.observed, self.mask, [[30., 32.]], [[32., 32.]],
                          max_displacement_fraction=1.)


if __name__ == '__main__':
    unittest.main()
