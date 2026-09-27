# Published reference-guided inpainting baseline — 27 September 2026

This reproduces an external author model locally; it is not a new project contribution. Four previously observed development cases, four separately supplied reference choices per case, native 256 inference and exact outside-mask 512 composition. Reference 0 was fixed as the primary choice before generation; no best-of-four selection.

16/16 new predictions; 12 reused controls; 28/28 scored rows. All 28 successful rows preserve known pixels. 3 detector-failure events are retained in JSON. Means show their complete identity denominators; scores are not accuracy percentages.

| Arm | FaceNet ↑ (n) | Gallery ↑ (n) | Hole MAE ↓ (n) | LPIPS ↓ (n) |
|---|---:|---:|---:|---:|
| reference_0 | 0.519579 (4) | 0.345113 (4) | 0.092307 (4) | 0.043791 (4) |
| reference_1 | 0.536569 (4) | 0.292262 (4) | 0.096926 (4) | 0.044149 (4) |
| reference_2 | 0.503230 (4) | 0.266759 (4) | 0.090860 (4) | 0.043514 (4) |
| reference_3 | 0.534709 (4) | 0.325191 (4) | 0.096383 (4) | 0.043093 (4) |
| scaffold | 0.726530 (4) | 0.400028 (4) | 0.063200 (4) | 0.025153 (4) |
| reference_context | 0.776886 (4) | 0.451593 (4) | 0.065539 (4) | 0.027261 (4) |
| observed | 0.480200 (3) | 0.183254 (3) | 0.197037 (4) | 0.098775 (4) |

| Prespecified paired contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |
|---|---:|---:|---|---:|
| reference_0_minus_scaffold/facenet_cosine | 4 | -0.206952 | [-0.299235463142395, -0.1026577353477478] | 0.500000 |
| reference_0_minus_scaffold/hole_mae | 4 | 0.029106 | [0.019410694790688895, 0.04139310478171852] | 0.500000 |
| reference_0_minus_reference_context/facenet_cosine | 4 | -0.257308 | [-0.34632508456707, -0.17594455182552338] | 0.500000 |
| reference_0_minus_reference_context/hole_mae | 4 | 0.026768 | [0.01703087645477941, 0.03771195445740172] | 0.500000 |

Four tests were specified before generation; identity is the statistical unit. Four references are not four independent people, and deterministic inference is not replicated diffusion sampling. With four identities, the smallest two-sided exact p is 0.125; bootstrap intervals are unstable. Conditioning reference count differs from the four-reference ReF-LDM comparator. No equal-compute or pretraining-disjoint claim is made.

Mean new generation time including reference parsing, conversion and saving is 0.132s (model loading excluded); peak allocated GPU memory is 0.908GiB. These are this laptop's screen measurements, not a general speed benchmark.

Both author model and author ArcFace R101 plus BiSeNet reference parser were strict-loaded. An adapter check matches the author native inference method bit-for-bit on synthetic inputs. Only reference pixels enter parsing; no target truth, target parsing, target identity feature or gallery enters generation. Eight reserved final identities remain unopened. Native256 synthesis is upsampled, not a native512 model.

Official sources: [paper](https://arxiv.org/abs/2303.07014), [author code](https://github.com/WuyangLuo/RefFaceInpainting), [reference parser](https://github.com/zllrunning/face-parsing.PyTorch). Exact commits, weight hashes, frozen inputs and numerical evidence are retained. See `REFFACE_BASELINE_REVIEW_V1.md` for visual findings and the resulting research decision.
