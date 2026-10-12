# Masked SSIM capacity audit and conditional next experiment

12 October 2026. This document records a source/evidence audit and one conditional research proposal. It is **not an approved or frozen experiment protocol**. No adapter implementation, model modification, training, generation or development-score tuning was performed for this audit. Only this report was added. The current SSIM continuation must finish under its existing protocol; a future capacity experiment would require its own saved protocol and independent feasibility probe.

## Current trainable path

`scripts/sdxl_token_compatibility_probe_v1.py::CompatibilityScorer` is a shared `Linear(10,32) -> SiLU -> Linear(32,1)` network. Its exact registered parameter count is `(10*32+32)+(32*1+1) = 385`. Its output is `0.5*tanh(logit)`, an additive appearance-attention bias bounded to [-0.5,0.5].

`TokenCompatibilityProcessor` applies this bias at 70 existing cross-attention processors. Four reference sets supply 16 learned appearance tokens each, flattened jointly to 64 tokens. The learned bias reweights frozen appearance key/value projections; it does not train those projections, create new reference value features, or supply anatomical coordinates. Text and FaceID paths remain unchanged. Appearance output is restricted to hole queries at the processor, with the fixed branch scale 0.2; normalized attention cannot abstain from the entire appearance branch. Subsequent layers can propagate a local change, so processor locality does not guarantee raw-image locality.

The ten inputs are query/key cosine, normalized RMS difference, normalized channel mean/std differences, query/key log norms, norm difference, key consensus cosine, timestep fraction and visible-target fraction. Some are redundant or constant within a call: normalized RMS difference is a deterministic function of cosine and channel count, while timestep and visible fraction are shared across its token pairs. One network is shared across layers, heads and references. None of these features explicitly identifies iris, pupil, lid or brow locations.

The query/key feature graph is retained; this is not a detached-gradient bug. A different decoded objective can nonetheless be restricted to changing the same narrow set of appearance mixtures. The bounded bias changes the relative odds of two reference tokens by at most `exp(1)` relative to their original attention odds at that operation; it does not remove their frozen content basis. This is a direct property of the attention mechanism, not proof that its bound is saturated in actual training.

`scripts/train_masked_sdxl_ssim_continuation_v1.py` preserves 3,366,130,756 frozen loaded UNet/adapter parameters and a separate frozen CPU fp32 VAE with 83,653,863 parameters. Only the 385 scorer parameters train. It uses eight people, 32 updates, batch one, timesteps 100/300, training resolution 512, and evaluation resolution 1024. The decoded objective supervises a one-step clean-latent estimate from noised training targets; it does not backpropagate through the complete 30-step inference trajectory. These data, schedule and resolution limitations are separate from parameter capacity.

## Measured gradient evidence and its limits

Source: `outputs/masked_sdxl_ssim_loss_probe_v1/receipt.json`. The training-only probe uses identity 6792, timestep 300, noise seed 20261011 and the immutable 64-update starting scorer. It passed finite-gradient/update and frozen-weight guards.

| Measurement | Value |
|---|---:|
| Epsilon gradient norm | 0.00007034015288809314 |
| Unweighted SSIM gradient norm | 0.0007268317858688533 |
| SSIM gradient norm after fixed weight 0.1 | 0.00007268317858688533 |
| Unweighted SSIM / epsilon norm ratio | 10.333099318467475 |
| Weighted SSIM / epsilon norm ratio | 1.0333099318467475 |
| Epsilon/SSIM gradient cosine | 0.9968613402152373 |
| Combined gradient norm | 0.00014291108527686447 |
| Peak allocated GPU bytes | 7,337,905,664 |

The weighted structural gradient has substantial magnitude and is almost parallel to epsilon at this case. This supports a plausible explanation that different image objectives project onto nearly the same scorer update direction. It does **not** establish global gradient rank, behavior at other cases/timesteps, saturated attention, or capacity as the sole cause of poor eye geometry. It does not justify calling SSIM ineffective before the current matched evaluation finishes.

The earlier RGB training-only probe recorded cosine 0.9988125 between RGB and epsilon gradients. Its completed 16-update matched continuations had mean hole SSIM 0.4449920403 for epsilon and 0.4448602374 for RGB. The recorded raw/final visual review retained incorrect gaze, iris-colour artifacts and inaccurate eye/brow geometry. Those prior negative results motivate diagnosing the available update path; they were not used to select the proposed layer/rank by a development sweep. See `research/MASKED_SDXL_DECODED_CONTINUATION_RESULTS_V1.md` and `research/masked_sdxl_decoded_loss_probe_v1_evidence.json`.

Differentiable CPU VAE placement accounts for measured compute overhead. The nonzero audited gradients mean CPU placement itself is not evidence that the objective cannot reach the scorer. More GPU memory alone has not been established as a remedy.

## One conditional capacity experiment

If the current SSIM candidate fails its fixed gates, the single proposed next experiment is **rank-4 query/value residual adaptation at one late spatial self-attention layer**, with the output difference restricted to hole queries.

Exact frozen tensor paths:

- `unet.up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_q.weight`
- `unet.up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_v.weight`

The local pinned UNet safetensors header confirms both tensors are `[640,640]`. This is the last transformer in the highest-resolution attention up block: spatial attention operates on a 32x32 grid at 512 training and 64x64 at 1024 inference. Its native keys, output projection, convolutions, text conditioning, FaceID and appearance projections remain frozen.

For each of Q and V, add `A[4,640]` and `B[640,4]`, giving `Q' = Q + Bq(Aq(h))` and `V' = V + Bv(Av(h))`. Use fixed rank scaling 1 (equivalently alpha=rank=4), random A and zero B. Do not also zero-initialize a multiplicative gate: that would obstruct the initial adapter gradient.

| Parameter scope | Exact count |
|---|---:|
| Q low-rank pair | 4*640 + 640*4 = 5,120 |
| V low-rank pair | 4*640 + 640*4 = 5,120 |
| New parameters | 10,240 |
| Existing scorer, retained trainable | 385 |
| Candidate total trainable | 10,625 |
| Equal-loss control total trainable | 385 |

Preserve the original self-attention delegate output and add the adapted-minus-original attention result only at original hole queries. Both terms in this difference must use the same masked query shape and the same output-map convention, avoiding duplicate output bias. The new path should retain upstream activation gradients. Do not globally replace Q/V for visible queries.

The technical hypothesis is that the structural objective needs access to spatial feature routing and content, rather than only a pooled bias over fixed reference values. Q adaptation changes where a hole query retrieves target-context features; V adaptation changes the content those spatial features can contribute. The layer choice is architectural: it is the final high-resolution spatial attention layer, not a layer selected by development outcomes.

This intervention does **not** establish correct reference correspondence. It retains the same compressed reference conditioning and may learn a generic eye prior or overfit eight identities. It cannot guarantee recovery of the true missing gaze/expression. Unlike the failed spatial-reference proposal, it does not require constructing or trusting a new reference-to-target geometric alignment. That avoids that prerequisite, but does not solve correspondence.

## Fair comparison and data guards

A future saved protocol should compare the 385-parameter control with the 10,625-parameter candidate under the **same** `0.25 full epsilon MSE + 0.75 hole epsilon MSE + 0.1 masked SSIM loss`. Both should start from the original 64-update scorer SHA256 `0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba`; only the candidate's new B tensors start at zero. The current SSIM and epsilon final checkpoints must not be chosen according to favorable development scores as new starting points.

Keep the scorer trainable in both arms. Reset the same AdamW optimizer settings, preserve the ordered eight-person cache, four reference order, original masks, 32 updates, timesteps 100/300, matched noise, precision, decoder, reference scales and fixed inference/evaluation recipe. Isolate adapter initialization RNG so it cannot perturb the matched noise sequence. This is a capacity comparison, not another loss ablation. No rank/layer sweep, extra loss, alignment, reference selection, corruption change, early stopping or best-checkpoint selection is part of it.

Training identities remain 6792, 2250, 2083, 4164, 674, 476, 1281 and 8748. Exclude 263, 6681, 10149 and protected 7400, plus every existing exclusion. Verify training target/reference roles and source hashes, original cache/manifest/mask hashes, checkpoint hashes and model assets before optimization. Evaluation originals and annotations enter only the unchanged scoring/reporting stage; no hidden evaluation pixels or failed alignments enter features, losses or adapter setup. Preserve all failures and final checkpoints in new folders. The existing result gates remain mandatory; this report grants no training, multi-seed expansion or promotion.

## Independent probe and neutral replay requirements

Before any future training, save a separate protocol and source pins, then verify the exact parameter count and frozen-weight set. Synthetic checks should cover enabled-zero equality, finite nonzero B gradients, an effective disposable update and unchanged output outside the hole at the modified layer. Zero initial A gradients are expected when B is zero; a subsequent disposable step should verify A's gradient path. No shortcut that bypasses the adapter only when its weights are zero should substitute for testing the enabled path.

Preserving the original delegate plus identical-shape masked baseline/adapted calculations is intended to make the initial output difference exactly zero. This property must be measured on the actual precision/backend path, not assumed from algebra. Check original-scorer gradient agreement at the neutral state too, because the additional cancellation path must not silently alter the matched control.

A training-only disposable full-model backward on the existing original training case must then verify finite structural/epsilon/new-adapter gradients, nonzero parameter change, no frozen UNet/VAE gradients, restored backend flags and unchanged old checkpoint hashes. It creates no checkpoint or development quality output. Require an **enabled initial-state** 1024 inference replay to reproduce the saved 64-update raw and finished PNGs exactly before interpreting a capacity comparison. Retain raw images and check final visible pixels independently; local attention gating does not guarantee raw visible pixels remain unchanged downstream.

## Expected cost, not measured feasibility

The 10,240 extra fp32 parameters plus gradients and two Adam moment arrays occupy approximately 163,840 bytes, excluding framework overhead. The candidate adds two masked-query attention calculations plus low-rank projections at one layer; activation/backward cost matters much more than parameter storage. Its actual allocation peak and runtime have not been measured.

The prior RGB experiment measured 322.094 worker seconds for 16 decoded updates and 415.516 seconds for seven evaluation generation calls. Extrapolation suggests the same order of cost here: roughly 10-12 minutes per 32-update decoded arm, plus roughly seven minutes for seven matched generation calls, with additional scoring/setup time. New attention overhead and SSIM decoding can change this estimate. Two fresh equal-loss arms would therefore be a small controlled pilot, not full backbone fine-tuning.

The current SSIM probe already peaks at 7.338 decimal GB on the 8 GB laptop. A future protocol should retain a training-only resource probe, the 7.8 GB allocated-memory cap and a bounded worker, for example the current 1,800-second per-arm bound. Those are proposed limits, not evidence that the new path fits. Stop and report a resource failure rather than silently changing resolution, rank or data. No capacity benefit, geometry improvement or generalization claim follows from this audit.

## Outcome update after the fixed SSIM evaluation

The existing SSIM continuation subsequently completed all 32 updates and all three development reconstructions. It fails the combined numerical gate: mean hole SSIM is 0.444849 versus matched epsilon 0.444932 and current CFG3 0.454540; FaceNet falls to 0.782534 versus matched epsilon 0.785901. Small eye SSIM and LPIPS gains are retained in the full results. Visual inspection found no clear additional artifact at the displayed scales, but the original gaze/iris/brow errors remain. See [the complete results](MASKED_SSIM_CONTINUATION_RESULTS_V1.md).

The actual SSIM worker required **1,458.688 seconds (24.31 minutes)**, so the earlier RGB-based 10–12-minute extrapolation underestimated this run. Actual matched generation took 420.375 seconds. Future capacity cost must be measured independently; the extra path has not been shown to fit the remaining time/memory margin. The condition motivating this proposal is now met, but no capacity code, probe or training has been executed. It remains the next proposed single intervention, not an improved model.
