# Three-person eye fidelity: checkpoint scheduler diagnostic

9 October2026. Separate from the guidance3 experiment. The cached SDXL checkpoint specifies `EulerDiscreteScheduler`; the existing reference runner explicitly substitutes DDIM. Both are valid samplers, so this is not evidence of a wiring bug. Test whether the checkpoint's configured sampler improves the observed iris/brow fidelity on the same three development cases.

Change only DDIM to the checkpoint's original Euler scheduler. Keep guidance5, FaceID scale0.8,30steps,strength1,seed17,1024model/512output, exact eye masks, all four unchanged clean references and Poisson finishing. Do not combine guidance3 and Euler or choose settings per person. The configured timestep schedule/noise scaling changes as part of this sampler intervention; the seed label remains fixed, not every intermediate tensor.

Use the unchanged generator via a preloaded runtime holding the pipeline's original scheduler. Record scheduler config and source hashes. Store new outputs under `outputs/focused_eye_scheduler_v1`. No training, model replacement, input changes or website default changes.

Score with the same frozen regional evaluator against the same guidance5/DDIM control. Report hole PSNR/SSIM, hole/eye crop LPIPS, both identity metrics, every per-person regression and detector coverage. Require improved mean hole SSIM without worsening the remaining mean measures and inspect visual colour/gaze/brow fidelity before considering any promotion. Repeated development testing does not establish unknown-client accuracy. Clean targets remain evaluation-only.
