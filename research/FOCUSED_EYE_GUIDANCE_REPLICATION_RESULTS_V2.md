# Three-person lower-guidance replication

Fixed persons 263, 6681 and 10149; seeds 17, 29 and 43; four real clean references; identical erased-eye masks. DDIM 30, FaceID scale 0.8, strength 1, 1024 inference / 512 output and Poisson finishing. Only CFG 5 versus 3 changes. No training or website default changes. Six seed-17 outputs are hash-verified reused controls/candidates; twelve new GPU calls.

Generation: 18/18 complete. Every failed attempt remains in the source summary. All means below use identical valid pairs for each metric; coverage is explicit.

## Nine-pair means

| Metric | CFG 5 control | CFG 3 candidate | Delta | Valid pairs | Improved / worsened |
|---|---:|---:|---:|---:|---:|
| hole_psnr | 17.017847 | 17.821481 | +0.803634 | 9/9 | 9 / 0 |
| hole_ssim | 0.414146 | 0.433098 | +0.018952 | 9/9 | 8 / 1 |
| hole_lpips | 0.201441 | 0.191851 | -0.009590 | 9/9 | 8 / 1 |
| eye_lpips | 0.159577 | 0.144985 | -0.014592 | 9/9 | 8 / 1 |
| facenet_cosine | 0.769196 | 0.798866 | +0.029671 | 9/9 | 7 / 2 |
| arcface_conditioning_cosine | 0.760720 | 0.785079 | +0.024359 | 9/9 | 8 / 1 |

## Every person: three-seed means

| Person | Metric | Control | Candidate | Delta | Valid pairs |
|---|---|---:|---:|---:|---:|
| 263 | hole_psnr | 16.066807 | 16.780702 | +0.713894 | 3/3 |
| 263 | hole_ssim | 0.479746 | 0.505132 | +0.025386 | 3/3 |
| 263 | hole_lpips | 0.167959 | 0.157458 | -0.010501 | 3/3 |
| 263 | eye_lpips | 0.177055 | 0.167921 | -0.009134 | 3/3 |
| 263 | facenet_cosine | 0.727294 | 0.766180 | +0.038887 | 3/3 |
| 263 | arcface_conditioning_cosine | 0.778760 | 0.803343 | +0.024584 | 3/3 |
| 6681 | hole_psnr | 18.691516 | 19.782117 | +1.090601 | 3/3 |
| 6681 | hole_ssim | 0.451021 | 0.471024 | +0.020003 | 3/3 |
| 6681 | hole_lpips | 0.159568 | 0.148377 | -0.011191 | 3/3 |
| 6681 | eye_lpips | 0.132977 | 0.113563 | -0.019414 | 3/3 |
| 6681 | facenet_cosine | 0.794844 | 0.858587 | +0.063743 | 3/3 |
| 6681 | arcface_conditioning_cosine | 0.769596 | 0.813014 | +0.043417 | 3/3 |
| 10149 | hole_psnr | 16.295218 | 16.901625 | +0.606408 | 3/3 |
| 10149 | hole_ssim | 0.311671 | 0.323138 | +0.011467 | 3/3 |
| 10149 | hole_lpips | 0.276794 | 0.269717 | -0.007077 | 3/3 |
| 10149 | eye_lpips | 0.168697 | 0.153471 | -0.015226 | 3/3 |
| 10149 | facenet_cosine | 0.785449 | 0.771830 | -0.013618 | 3/3 |
| 10149 | arcface_conditioning_cosine | 0.733803 | 0.738879 | +0.005077 | 3/3 |

## Every seed: three-person means

| Seed | Metric | Control | Candidate | Delta | Valid pairs |
|---|---|---:|---:|---:|---:|
| 17 | hole_psnr | 17.019171 | 17.894556 | +0.875386 | 3/3 |
| 17 | hole_ssim | 0.428353 | 0.454540 | +0.026187 | 3/3 |
| 17 | hole_lpips | 0.193782 | 0.178454 | -0.015328 | 3/3 |
| 17 | eye_lpips | 0.171193 | 0.152351 | -0.018842 | 3/3 |
| 17 | facenet_cosine | 0.764280 | 0.784486 | +0.020206 | 3/3 |
| 17 | arcface_conditioning_cosine | 0.787376 | 0.816156 | +0.028780 | 3/3 |
| 29 | hole_psnr | 17.264975 | 18.060327 | +0.795352 | 3/3 |
| 29 | hole_ssim | 0.414474 | 0.434709 | +0.020235 | 3/3 |
| 29 | hole_lpips | 0.191332 | 0.179318 | -0.012014 | 3/3 |
| 29 | eye_lpips | 0.149736 | 0.136791 | -0.012945 | 3/3 |
| 29 | facenet_cosine | 0.763171 | 0.794259 | +0.031087 | 3/3 |
| 29 | arcface_conditioning_cosine | 0.754818 | 0.779686 | +0.024868 | 3/3 |
| 43 | hole_psnr | 16.769395 | 17.509560 | +0.740165 | 3/3 |
| 43 | hole_ssim | 0.399611 | 0.410044 | +0.010433 | 3/3 |
| 43 | hole_lpips | 0.219208 | 0.217781 | -0.001427 | 3/3 |
| 43 | eye_lpips | 0.157801 | 0.145813 | -0.011988 | 3/3 |
| 43 | facenet_cosine | 0.780135 | 0.817853 | +0.037718 | 3/3 |
| 43 | arcface_conditioning_cosine | 0.739965 | 0.759394 | +0.019429 | 3/3 |

## Every matched image

| Person | Seed | Metric | Control | Candidate | Delta |
|---|---:|---|---:|---:|---:|
| 263 | 17 | hole_psnr | 16.413578 | 17.429831 | +1.016253 |
| 263 | 17 | hole_ssim | 0.494264 | 0.538117 | +0.043853 |
| 263 | 17 | hole_lpips | 0.168389 | 0.144067 | -0.024322 |
| 263 | 17 | eye_lpips | 0.206952 | 0.175436 | -0.031516 |
| 263 | 17 | facenet_cosine | 0.728110 | 0.750827 | +0.022717 |
| 263 | 17 | arcface_conditioning_cosine | 0.802763 | 0.821858 | +0.019095 |
| 263 | 29 | hole_psnr | 16.150216 | 16.889313 | +0.739097 |
| 263 | 29 | hole_ssim | 0.474352 | 0.513830 | +0.039478 |
| 263 | 29 | hole_lpips | 0.172252 | 0.155881 | -0.016371 |
| 263 | 29 | eye_lpips | 0.178979 | 0.170976 | -0.008003 |
| 263 | 29 | facenet_cosine | 0.718604 | 0.761331 | +0.042727 |
| 263 | 29 | arcface_conditioning_cosine | 0.760094 | 0.795215 | +0.035121 |
| 263 | 43 | hole_psnr | 15.636628 | 16.022961 | +0.386334 |
| 263 | 43 | hole_ssim | 0.470623 | 0.463449 | -0.007174 |
| 263 | 43 | hole_lpips | 0.163236 | 0.172427 | +0.009191 |
| 263 | 43 | eye_lpips | 0.145234 | 0.157351 | +0.012117 |
| 263 | 43 | facenet_cosine | 0.735167 | 0.786384 | +0.051216 |
| 263 | 43 | arcface_conditioning_cosine | 0.773423 | 0.792958 | +0.019535 |
| 6681 | 17 | hole_psnr | 18.910703 | 19.859689 | +0.948986 |
| 6681 | 17 | hole_ssim | 0.493393 | 0.512666 | +0.019273 |
| 6681 | 17 | hole_lpips | 0.147842 | 0.136960 | -0.010883 |
| 6681 | 17 | eye_lpips | 0.122778 | 0.118169 | -0.004610 |
| 6681 | 17 | facenet_cosine | 0.766272 | 0.840004 | +0.073732 |
| 6681 | 17 | arcface_conditioning_cosine | 0.786206 | 0.840881 | +0.054675 |
| 6681 | 29 | hole_psnr | 18.814127 | 19.953650 | +1.139523 |
| 6681 | 29 | hole_ssim | 0.443863 | 0.460066 | +0.016203 |
| 6681 | 29 | hole_lpips | 0.148180 | 0.134545 | -0.013635 |
| 6681 | 29 | eye_lpips | 0.119587 | 0.103195 | -0.016392 |
| 6681 | 29 | facenet_cosine | 0.771809 | 0.829692 | +0.057883 |
| 6681 | 29 | arcface_conditioning_cosine | 0.766668 | 0.791210 | +0.024542 |
| 6681 | 43 | hole_psnr | 18.349719 | 19.533011 | +1.183292 |
| 6681 | 43 | hole_ssim | 0.415807 | 0.440340 | +0.024533 |
| 6681 | 43 | hole_lpips | 0.182682 | 0.173626 | -0.009055 |
| 6681 | 43 | eye_lpips | 0.156567 | 0.119325 | -0.037242 |
| 6681 | 43 | facenet_cosine | 0.846451 | 0.906066 | +0.059615 |
| 6681 | 43 | arcface_conditioning_cosine | 0.755915 | 0.806950 | +0.051035 |
| 10149 | 17 | hole_psnr | 15.733232 | 16.394149 | +0.660917 |
| 10149 | 17 | hole_ssim | 0.297403 | 0.312838 | +0.015435 |
| 10149 | 17 | hole_lpips | 0.265115 | 0.254336 | -0.010779 |
| 10149 | 17 | eye_lpips | 0.183849 | 0.163447 | -0.020401 |
| 10149 | 17 | facenet_cosine | 0.798458 | 0.762628 | -0.035830 |
| 10149 | 17 | arcface_conditioning_cosine | 0.773159 | 0.785730 | +0.012571 |
| 10149 | 29 | hole_psnr | 16.830582 | 17.338019 | +0.507437 |
| 10149 | 29 | hole_ssim | 0.325208 | 0.330232 | +0.005024 |
| 10149 | 29 | hole_lpips | 0.253562 | 0.247526 | -0.006036 |
| 10149 | 29 | eye_lpips | 0.150640 | 0.136201 | -0.014439 |
| 10149 | 29 | facenet_cosine | 0.799100 | 0.791753 | -0.007348 |
| 10149 | 29 | arcface_conditioning_cosine | 0.737692 | 0.752634 | +0.014942 |
| 10149 | 43 | hole_psnr | 16.321840 | 16.972708 | +0.650869 |
| 10149 | 43 | hole_ssim | 0.312402 | 0.326343 | +0.013941 |
| 10149 | 43 | hole_lpips | 0.311706 | 0.307289 | -0.004417 |
| 10149 | 43 | eye_lpips | 0.171602 | 0.160763 | -0.010839 |
| 10149 | 43 | facenet_cosine | 0.758787 | 0.761111 | +0.002323 |
| 10149 | 43 | arcface_conditioning_cosine | 0.690557 | 0.678274 | -0.012283 |

## Numerical decision

Combined mean gate: PASS.
- Person 263, three-seed gate: PASS.
- Person 6681, three-seed gate: PASS.
- Person 10149, three-seed gate: FAIL.

These gates do not authorize universal promotion. Visual eye/gaze/brow fidelity and per-image identity regressions remain separate. This is repeated development on three people, not nine independent people, a held-out test, population accuracy, or an iris-colour percentage. ArcFace also conditions generation; FaceNet is separate. Anatomical eye crops and all detector records remain in the frozen score files.

Preflight: FOCUSED_EYE_GUIDANCE_REPLICATION_PREFLIGHT_V2.md. Summary and full provenance: outputs/focused_eye_guidance_replication_v2.

## Visual review and next intervention

All nine matched control/candidate images were inspected in the three contact sheets. Neither the mean gate nor higher whole-face cosine establishes correct iris texture or gaze. The setting remains experimental and is not promoted universally.
- Person 263: Lower guidance usually softens exaggerated contrast, but iris colour and gaze remain different from the original across all three seeds.
- Person 6681: Lower guidance reduces oversaturation in some outputs and improves mean identity; brow/eye structure still varies with seed.
- Person 10149: Iris/gaze and eyebrow-shape errors persist across all seeds; mean FaceNet drops 0.013618 despite higher mean SSIM.

Stop tuning the guidance setting on these same targets. Test a separate visual-reference appearance branch at fixed scale, with a neutral byte-exact replay and otherwise unchanged controls. See FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md. No appearance results are claimed in this guidance report.
