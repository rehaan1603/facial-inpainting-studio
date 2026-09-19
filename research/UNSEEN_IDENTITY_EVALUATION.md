# Local unseen-identity evaluation

The machine-readable authority is `protocols/unseen_identity_protocol_v1.json`. It records source hashes, every exclusion source, detector-screening attempts and explicit target/reference/gallery roles.

All identity labels in the reviewed mask-refiner training partition are excluded, along with prior pilot/benchmark/assessment/object-test cases and every attempted identity in the earlier reference diagnostic screen. Four validation groups have eight accepted photos each: one target, four runtime references and three withheld evaluation-gallery photos. Eight further official-test groups are reserved with the same roles. They are not used for generation or scorer tuning in this cycle.

Selection uses a fixed hash ranking and no generated performance. Validation photographs must have one detectable face. Reserved-final selection checks hashes and conservative near-duplicate fingerprints without face-content screening. Source/decoded hashes and pHash distance greater than six screen overlap with protected prior identities and retained roles/splits. pHash is not proof of independent captures or absence of every crop/flip duplicate. Dataset identity annotations are not independently verified.

The validation images were locally unexposed at preparation; once their results are inspected they become development evidence. They must never be relabeled fresh final data. The final identity reservation should also be respected by future practice-folder exports. LaPa has no verified corresponding identity mapping and is excluded from this identity-disjoint claim.

Pretrained SDXL, FaceID, LaMa and recognition encoders may have encountered these people or images. This protocol establishes separation from recorded **local** development only. It cannot establish pretraining-independent or population-wide generalization.

Statistical unit: identity, with condition/seed pairs averaged within identities. Primary endpoint: FaceNet target cosine, four mask-aware comparisons corrected by Holm. Gallery cosine, conditioning-encoder ArcFace, NIQE, BRISQUE, LPIPS, SSIM and pixel errors are complementary exploratory evidence. Keep detector failures and paired coverage explicit.
