# Three-face reference-strength probe

9 October2026. User requested quality improvement on three fixed faces before broader work.

The1024 + Poisson candidate improves average hole metrics and identity over512 hard composition, but gaze/iris appearance still differs. Test ONE fixed reduction in FaceID scale,0.8→0.4, to assess whether excessive reference conditioning dominates the visible target context. This is a hypothesis, not an established cause.

Identities263,6681,10149; eyes only; four clean references in their original order; seed17;1024 processing;30 DDIM steps; guidance5; strength1; same prompt, neutralized inputs and Poisson blending. Save new outputs in `outputs/focused_reference_strength_v1`; never overwrite controls. No training or reference substitution.

Compare all three pairs with the unchanged1024 blended scale0.8 controls. Require lower hole/eye crop LPIPS, no identity degradation, and no hole PSNR/SSIM regression. Report every per-person change and inspect gaze/shape. If metrics trade off or the images remain wrong, retain the tested control and record the negative result; do not silently cherry-pick different scales per person. These three development cases cannot establish generalization or calibrated accuracy.
