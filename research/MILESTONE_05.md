> Historical stage report. Later training, evaluation and duplicate-review status is in PROJECT_STATUS.md and FINAL_RESULTS.md; near_duplicate_review_v2.json supersedes the partial review.

# Duplicate screening and pretraining exposure

The read-only screening covered all 51,944 images in the current exact-file-cleaned manifests. It found 496 cross-partition or cross-dataset perceptual-hash candidate pairs within Hamming distance 6. None had identical decoded RGB arrays. That does not rule out recompressed, resized or lightly altered copies.

## Visual triage and separate manifests

Forty-two pairs have been visually reviewed: the 32 pairs with lowest resized RGB error and every flagged pair touching the 96-identity v2 benchmark. Eleven pairs were judged near-duplicate photographs and 31 were visibly different photographs. These are photographic comparisons, not identity-recognition decisions. The remaining 454 candidates are unreviewed, not assumed safe.

All 22 image files participating in the reviewed near-duplicate pairs are excluded from separate provisional manifests:

| Manifest | Before | Retained | Excluded |
|---|---:|---:|---:|
| celebahq_reviewed_v1.csv | 29,946 | 29,926 | 20 |
| lapa_reviewed_v1.csv | 21,998 | 21,996 | 2 |

No original image, annotation or prior manifest was modified. Historical experiments retain their original hashes. None of the 22 exclusions occurs among the cases in pilot_clean_v1, benchmark_v2 or area_matched_v3_margin12. This means these particular exclusions do not invalidate those selected cases; it does not establish complete contamination independence.

Candidate IDs, paired source paths, visual decisions, manifest hashes and case-overlap checks are stored in near_duplicate_review.json. The review-writing script refuses to apply its decisions if the candidate-index hash changes. Future runs must explicitly choose the reviewed manifests; historical scripts intentionally retain their original inputs.

## What remains unresolved

- Review the other 454 candidate pairs and obtain independent verification of exclusion decisions before publication.
- Extend screening to crops, flips and transformed images that a single global perceptual hash can miss. Within-dataset within-split near duplicates were not output by this screen.
- Distinguish photograph duplication from shared identities. Identity-label separation alone does not prove actual identity independence, and this audit performs no recognition.
- Obtain or document missing pretrained-model training manifests. PRETRAINING_EXPOSURE_LEDGER.md records current evidence for LaMa, ResShift, its VAE and LPIPS.

The audit inspected test images for integrity only. Test images have not been used for model inference or setting selection. The perceptual-hash candidate search passed an exact comparison against brute-force search on a test collection, including distances 0 through 6. Passing that algorithm test does not establish photographic recall or precision.

## Reproduction

Run scripts/audit_near_duplicates.py, scripts/rank_duplicate_candidates.py, and scripts/save_duplicate_review.py using the project environment. Fingerprints are cached with source-file and manifest provenance. Repeating the visual decisions requires the same candidate-index hash. Review sheets are in outputs/near_duplicate_audit; they are local inspection artifacts and are not licensed for public redistribution merely because they were generated here.

The next research work remains realistic held-out occluder construction and finishing contamination review before substantial training. This milestone improves data hygiene; it is not a novel inpainting method.
