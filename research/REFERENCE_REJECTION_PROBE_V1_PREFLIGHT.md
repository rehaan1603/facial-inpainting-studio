# Reference rejection: causal diagnostic

5 October 2026, recorded before generation. Existing dataset only.

The expanded RGB-encoding check improves mean identity similarity but worsens LPIPS and eye LPIPS. Keep production unchanged. Before adding a learned rejection gate, test whether removing unreliable reference information is beneficial under the original matched local protocol.

## Fixed protocol

Use the four previously observed development identities 263, 6681, 7400 and 10149, seeds 17/29/43, frozen EMA ReF-LDM, 512 pixels, four separately encoded reference latents, 50 DDIM uniform-trailing steps, guidance 1.5, bfloat16, original rounding conversion and cached degraded inputs. No new training, synthetic references, targets, or reserved-final identity access.

Controls: hash-verified archived baseline clean and wrong-patch outputs (24 images). First reproduce both conditions for every identity at seed17 (eight extra calls), requiring byte-exact equality before using the controls for that identity.

Two new interventions, 12 images each:

1. `drop_first`: remove the first reference, which is the deliberately corrupted reference in the wrong-patch protocol. Keep the other three references unchanged. Compare with both original conditions. Under clean references this is an arbitrary removal control; under wrong-patch it is oracle knowledge of corruption location. It is not an inference-ready detection mechanism or an unbiased estimate of a learned selector.
2. `zero_references`: zero all four conditioning reference latents, matching the author's CFG null-reference convention. This retains the zero-reference cache path, so do not describe it as removing reference-attention tokens.

The same intervention images legitimately serve both baseline comparisons because neither reads the changed first reference. Count 24 fresh intervention images plus eight replay calls, not 48 distinct intervention outputs. Record hashes, timing and memory.

## Metrics and decision

Use existing frozen PSNR, SSIM, LPIPS, FaceNet, ArcFace and annotated eye-crop LPIPS, with paired available support and retained detector failures. Average seeds within each identity for interpretation. Compare removal against both the four-reference baseline and zero-reference control; do not call a gain over a deliberately damaged baseline alone novelty.

If oracle removal fails the joint identity/perceptual/eye requirement, do not train a gate on the assumption that dropping the bad reference solves fidelity. If it helps, the next required test is whether input-only evidence can identify that reference without reading corruption labels or clean target images. Report clean-reference cost and seed consistency. No automatic website promotion.
