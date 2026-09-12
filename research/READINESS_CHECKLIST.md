# Publication readiness after implementation

This checklist separates verified local work from unresolved publication requirements. Historical planning is retained in READINESS_CHECKLIST_INITIAL.md. Completion of engineering is not acceptance evidence.

- [x] Local CUDA environment and two working inpainting backbones.
- [x] Full supplied-image and annotation integrity audits.
- [x] Exact-file duplicate quarantine and review of all 496 pHash candidate pairs.
- [x] Identity-label-separated HQ development tuning and assessment.
- [x] Paired regional metrics, perceptual metric and analytic leakage checks.
- [x] Dilation/feather grid and matched-area development controls.
- [x] Matched generic and weighted segmentation training with three seeds.
- [x] Threshold selection restricted to tuning data.
- [x] Frozen object-test protocol with an external face-image source.
- [x] Local inference interface, versioned configurations and checkpoint hashes.
- [ ] Confirm scientific originality against the closest complete papers and executable methods.
- [ ] Add missing robust-mask and learned-alpha controls and convergence evidence.
- [ ] Establish a practically meaningful improvement with a prospectively chosen margin.
- [ ] Independently verify transformed duplicates, identities and pretraining memberships.
- [ ] Provide real captured occlusion evidence for any real-world claim.
- [ ] Add identity or human evaluation if those claims are made.
- [ ] Select the actual venue and complete human authorship, reference, attribution, formatting and disclosure review.
- [ ] Obtain peer-review acceptance; no submission has been made.

Final artifact checks are recorded in completion_checks.json. Exact test results and intervals are in FINAL_RESULTS.md. Neither a runnable demo nor an automatically generated manuscript closes the scientific items above.
