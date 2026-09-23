# Unfamiliar development functionality check — 23 September 2026

Two identities absent from all previous local selected/attempted groups: 1590 and 1529. Hash-ranked selection, duplicate filtering, eight photographs per identity: one target, four references and three withheld gallery images. All reserved-final identities excluded before pixel access. These identities are now observed development data. Pretraining overlap is unknown.

Two conditions per identity: medium central-face Gaussian blur and mixed damage. Four reference-restoration runs were scheduled. Three completed initially; 1590_mixed terminated with native process exit code 3221227274 and no Python traceback. The cause is unresolved. An unchanged-settings retry succeeded; the original failure is retained. Initial scoring is 10/12 rows; after separately recorded recovery it is 12/12. This includes unchanged controls and native/composed variants, not twelve independent subjects.

| Arm, after runtime recovery | FaceNet | Hole MAE | LPIPS |
|---|---:|---:|---:|
| observed | 0.9608506262302399 | 0.039907563893560156 | 0.028567225323058665 |
| native | 0.9528198540210724 | 0.047769256780650714 | 0.20336369797587395 |
| composited | 0.963497519493103 | 0.047769256780650714 | 0.01511677133385092 |

No statistical superiority or unknown-person reliability claim follows from two identities. This verifies operation beyond the original examples, with a retained runtime failure. Full metric coverage and provenance are in the JSON receipts.
