# Regional routing v2 — development results

Four previously observed development identity groups; three conditions; two seeds; six policies. 144 logical rows, with 24 original concatenation controls reused by verified hashes. No reserved-final images used; no training performed.

| Policy | FaceNet target ↑ | FaceNet gallery ↑ | NIQE ↓ | BRISQUE ↓ | Hole MAE ↓ | Valid target / 24 |
|---|---:|---:|---:|---:|---:|---:|
| concat | 0.8180 | 0.5304 | 4.7119 | 10.1203 | 0.0900 | 24 |
| equal | 0.8105 | 0.5270 | 4.7055 | 10.1391 | 0.0903 | 24 |
| identity | 0.8095 | 0.5203 | 4.7014 | 10.1335 | 0.0901 | 24 |
| quality | 0.8132 | 0.5219 | 4.7061 | 10.1419 | 0.0895 | 24 |
| mask_aware | 0.8112 | 0.5217 | 4.7098 | 10.1349 | 0.0899 | 24 |
| regional | 0.8137 | 0.5237 | 4.7093 | 10.1269 | 0.0899 | 24 |

| Regional minus | Identities | Pairs | Mean delta | Bootstrap 95% interval | Exact p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| concat | 4 | 24 | -0.0043 | [-0.014050995310147604, 0.005369638403256735] | 0.5000 | 1.0000 |
| equal | 4 | 24 | 0.0032 | [-0.0013272563616434736, 0.008168496191501617] | 0.3750 | 1.0000 |
| identity | 4 | 24 | 0.0041 | [-0.0030026187499364214, 0.011299545566240948] | 0.5000 | 1.0000 |
| quality | 4 | 24 | 0.0005 | [-0.0036445955435434976, 0.0035220384597778325] | 0.8750 | 1.0000 |
| mask_aware | 4 | 24 | 0.0025 | [-0.00106112410624822, 0.005128105481465658] | 0.3750 | 1.0000 |

This small reused development sample cannot establish unknown-identity generalization or superiority. Bootstrap intervals with four identities are unstable. Whole-image metrics can conceal local distortions. The numerical JSON retains condition means and coverage for all metrics.

Equal routing averages separately normalized attention outputs; original concatenation normalizes across all reference tokens jointly. The original concatenation control is therefore essential. All routing arms retain four references and equal weights outside damage. Regional routing uses global FaceID descriptors, not local image patches or learned spatial correspondence. It is built on existing Diffusers mask functionality and is not a novelty claim.

Every generated output and all failures remain local under outputs/regional_routing_v2. Numerical evidence can be published without dataset photographs. Broader development validation, actual visibility estimation, resolution calibration, spatial reference features, external baselines and final evaluation remain outstanding.
