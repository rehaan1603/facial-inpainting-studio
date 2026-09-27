import unittest
import torch
from src.preservation.reference_context_v2 import VisibleContextCorrector, sample_explicit


class FakeModel:
    parameterization = 'eps'
    alphas_cumprod = torch.tensor([.4, .8])
    def decode_first_stage(self, x): return x
    def encode_first_stage(self, x): return x
    def get_first_stage_encoding(self, x): return x


class ReferenceContextTests(unittest.TestCase):
    def setUp(self):
        self.observed = torch.full((1, 3, 8, 8), .25)
        self.missing = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
        self.missing[:, :, 2:6, 2:6] = True
        self.eps = torch.full_like(self.observed, .1)
        self.x = torch.full_like(self.observed, .2)
        self.t = torch.tensor([1])

    def test_disabled_hook_is_bit_exact(self):
        c = VisibleContextCorrector(self.observed, self.missing, gain=0, steps=[0])
        self.assertIs(c.modify_score(FakeModel(), self.eps, self.x, self.t, {}), self.eps)
        self.assertEqual(c.applied, [])

    def test_missing_input_pixels_do_not_affect_correction(self):
        changed = self.observed.clone()
        changed.masked_fill_(self.missing, -.8)
        a = VisibleContextCorrector(self.observed, self.missing, steps=[0])
        b = VisibleContextCorrector(changed, self.missing, steps=[0])
        torch.testing.assert_close(a.modify_score(FakeModel(), self.eps, self.x, self.t, {}),
                                   b.modify_score(FakeModel(), self.eps, self.x, self.t, {}), rtol=0, atol=0)

    def test_returned_epsilon_changes_clean_estimate_toward_visible_evidence(self):
        c = VisibleContextCorrector(self.observed, self.missing, gain=.5, steps=[0])
        out = c.modify_score(FakeModel(), self.eps, self.x, self.t, {})
        a = FakeModel.alphas_cumprod[self.t].reshape(-1, 1, 1, 1)
        before = (self.x - (1-a).sqrt()*self.eps)/a.sqrt()
        after = (self.x - (1-a).sqrt()*out)/a.sqrt()
        expected = torch.where(self.missing, before, .5*before+.5*self.observed)
        torch.testing.assert_close(after, expected)
        self.assertEqual(c.applied[0]['step_index'], 0)
        self.assertIs(c.modify_score(FakeModel(), self.eps, self.x, self.t, {}), self.eps)

    def test_explicit_sampler_actually_forwards_hook(self):
        calls = {}
        class Sampler:
            def make_schedule(self, **kwargs): calls['schedule'] = kwargs
            def ddim_sampling(self, *args, **kwargs): calls['sampling'] = kwargs; return 'result'
        c = object()
        result = sample_explicit(Sampler(), {}, {}, [3, 8, 8], self.x, c)
        self.assertEqual(result, 'result')
        self.assertIs(calls['sampling']['score_corrector'], c)
        self.assertEqual(calls['sampling']['corrector_kwargs'], {})
        self.assertEqual(calls['schedule']['ddim_num_steps'], 50)


if __name__ == '__main__': unittest.main()
