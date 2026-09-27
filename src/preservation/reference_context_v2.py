"""Experimental visible-context correction for a frozen epsilon DDIM sampler.

The difference of encodings isolates the observation replacement from VAE
round-trip bias. It does not ensure locality in latent space or hidden fidelity.
"""
import torch


class VisibleContextCorrector:
    def __init__(self, observed, missing, gain=.5, steps=(46, 47, 48, 49)):
        if observed.ndim != 4 or observed.shape[1] != 3 or missing.shape != (observed.shape[0], 1, *observed.shape[2:]):
            raise ValueError('Expected matching BCHW RGB and B1HW missing mask')
        if not torch.isfinite(observed).all() or observed.min() < -1 or observed.max() > 1:
            raise ValueError('Observation must be finite normalized RGB')
        if not 0 <= gain <= 1 or any(type(i) is not int or i < 0 for i in steps):
            raise ValueError('Invalid gain/step schedule')
        self.observed = observed.detach().clone()
        self.missing = missing.bool().detach().clone()
        self.gain, self.steps = gain, frozenset(steps)
        self.calls, self.applied = 0, []

    @torch.no_grad()
    def modify_score(self, model, epsilon, noisy, timestep, condition, **kwargs):
        index = self.calls
        self.calls += 1
        if self.gain == 0 or index not in self.steps:
            return epsilon
        if model.parameterization != 'eps':
            raise ValueError('Requires an epsilon-predicting model')
        alpha = model.alphas_cumprod[timestep].reshape(-1, 1, 1, 1).to(noisy)
        if (alpha <= 0).any() or (alpha >= 1).any():
            raise ValueError('Invalid diffusion alpha')
        sigma = (1 - alpha).sqrt()
        clean = (noisy - sigma * epsilon) / alpha.sqrt()
        decoded = model.decode_first_stage(clean).clamp(-1, 1)
        projected = torch.where(self.missing, decoded, self.observed)
        encode = lambda x: model.get_first_stage_encoding(model.encode_first_stage(x))
        delta = encode(projected) - encode(decoded)
        corrected = clean + self.gain * delta
        result = (noisy - alpha.sqrt() * corrected) / sigma
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite context correction')
        self.applied.append({'step_index': index, 'timestep': int(timestep[0]),
                             'latent_delta_abs_mean': float(delta.abs().mean()),
                             'visible_error_before': float((decoded-self.observed).abs().masked_select(~self.missing).mean())})
        return result


def sample_explicit(sampler, condition, null_condition, shape, noise, corrector,
                    steps=50, guidance=1.5):
    """Call the author sampling loop with an explicit empty corrector-options dict."""
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0., verbose=False)
    return sampler.ddim_sampling(condition, (1, *shape), x_T=noise,
        score_corrector=corrector, corrector_kwargs={}, unconditional_guidance_scale=guidance,
        unconditional_conditioning=null_condition)
