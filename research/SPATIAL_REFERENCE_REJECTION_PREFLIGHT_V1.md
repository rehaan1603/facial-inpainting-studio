# Spatial reference rejection: fixed feasibility protocol

6 October 2026. Recorded before generation. Existing development identities only.

Whole-reference removal improved mean identity similarity but worsened LPIPS. Test whether suppressing only reference-attention keys in the known corrupted region retains useful detail. No training, website change or final-test access.

## Fixed design

Use identities 263, 6681, 7400, 10149, seeds 17/29/43, clean and wrong_patch references. Preserve the original local protocol: frozen EMA ReF-LDM, separately encoded reference latents, 512 RGB, 50 DDIM uniform_trailing steps, guidance 1.5, bfloat16 and original uint8 rounding. Reuse 24 hash-verified baseline outputs. Require eight neutral seed-17 replays to match exactly, one per identity/condition.

Generate 48 intervention outputs: two policies x four identities x three seeds x two conditions.

* `oracle_region`: forbid attention to the central quarter of reference 1's spatial key/value tokens, corresponding to the deliberately replaced central 256-square image patch. Apply to all conditional target queries in all compatible attention layers. Self-attention, other references and the null CFG branch remain unchanged.
* `random_corner`: mask one equal-area corner of reference 1, selected by a fixed SHA-256 identity ranking before viewing outputs. This is a random-corner spatial control, not a uniform random box over every position. It can partially overlap the corrupted region. Reuse the same corner across conditions and seeds.

Neutral policy calls original attention unchanged. Both active policies set selected conditional attention logits to negative infinity, retaining other tokens. No strength search or best-seed selection. Unit checks must establish neutral equality, CFG-null preservation, exact masked-key/value exclusion and equal token counts across policies.

## Interpretation boundaries

This is oracle location knowledge, not a learned detector. Encoder/denoiser receptive fields may spread corrupt information beyond the selected spatial tokens; these masks cannot establish removal of every corruption effect. Clean-reference runs measure the cost of dropping valid regional information. No ground-truth target pixels, identity scores or segmentation annotations enter inference.

## Evaluation and decision

Frozen PSNR, SSIM, LPIPS, FaceNet, ArcFace and annotated eye bounding-box LPIPS. Keep failed detections and denominators. Compare oracle versus baseline and random-corner in each condition, average seed deltas within identity, and retain every output. Report per-seed means and identity-cluster bootstrap intervals. Four previously observed identities provide development evidence only.

Progression requires lower LPIPS, non-worsening PSNR/SSIM/FaceNet/eye LPIPS across each condition and seed, plus complete ArcFace measurement and non-worsening ArcFace. Numerical tolerance 1e-6 is a reproducibility safeguard, not a perceptual non-inferiority margin. Also report oracle-versus-random results; a stronger intervention alone is not an established contribution. No automatic deployment even if the screen passes. A successful feasibility outcome must later be replaced by input-only localization and independently validated.
