# Spatial adapter prerequisite correction — 19 October 2026

## Finding
A bug in the prerequisite comparison, not an established model defect, blocked the proposed capacity experiment. Control/native epsilon code calculated `.75 * masked_sum / denominator`; candidate and actual training calculated `.75 * (masked_sum / denominator)`. Floating-point operation order differed. Equal observed loss scalars did not imply equal derivatives. V6 changes only those control/native parentheses; no tolerance, data, checkpoint, model or quality metric was relaxed.

V3 and V4 both failed at maximum scorer-gradient difference1.811786205507815e-7. V5 removed intervening structural backward and still failed, ruling that operation out for this case. These failed receipts are preserved. Earlier claims that requires-grad flags or the added graph explained the mismatch were confounded by loss operation order and are superseded.

## V6 full GPU prerequisite: passed
- Original vs enabled-zero predictions: exact.
- Native vs corrected control scorer epsilon gradients: exact (maximum difference0).
- Corrected control vs candidate epsilon gradients: exact (maximum difference0).
- Control vs candidate structural scorer gradients: exact (maximum difference0).
- Original tolerance unchanged: atol1e-7, rtol1e-4.
- Q.B/V.B receive finite nonzero epsilon and masked-SSIM gradients and both change in the disposable update. A gradients start at zero as expected with zero B.
- Frozen native UNet/VAE have no gradients; prior checkpoint/source checks pass; backend flags restored.
- 10,240 added parameters; 10,625 total autograd parameters.
- Peak CUDA allocated7,338,198,528bytes; supervisor73.0seconds; no OOM/timeout.
- Disposable in-memory update only: no trained checkpoint and no quality scores.

## 1024 inference replay: passed
Person263, seed17, original64 scorer, same four references and inference settings. Both raw and finished PNGs are byte-exact and pixel-exact against the retained original64 output. Evaluation original was not read. Supervisor97.782seconds. This establishes zero-state compatibility on this case, not better reconstruction or unknown-image generalization.

## Next work
Implement and freeze the two-arm32-update experiment described in `SPATIAL_QV_MATCHED_TRAINING_PLAN_V6.md`. Keep all historical controls and produce six new development reconstructions; evaluate21 paired records total. Train only after source/optimizer/control-state guards are reviewed. The current website remains unchanged. No new SSIM/identity improvement is claimed. Multiple identities/seeds and corrupted-reference validation remain required after any promising development result.

Evidence: [Photo-free receipts and source hashes](spatial_qv_loss_order_evidence_v6.json).
