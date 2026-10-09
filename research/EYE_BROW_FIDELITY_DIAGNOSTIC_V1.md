# Supplementary eye and eyebrow fidelity diagnostics

Primary masked-region metrics and success criteria remain unchanged. This appendix responds to observed iris/gaze/brow errors: the larger erased rectangle includes substantial skin and is different from annotated anatomical eyes and eyebrows. Official target masks are used ONLY after inference for evaluation.

Pixel PSNR/MAE use only each annotated region inside the original missing mask. SSIM uses the same Gaussian sigma 1.5, 11-by-11, population-variance RGB map as the frozen evaluator, averaging only fully contained windows. Sparse or absent valid windows are disclosed; null SSIM is not replaced with zero. No iris-only pigmentation or gaze-angle accuracy is measured.

| Study | Person | Seed | Arm | Region | Pixel PSNR | MAE | SSIM | Valid centres | Pixels |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| guidance_replication | 263 | 17 | control | eyes | 11.324717 | 0.209358 | 0.189793 | 219 | 1349 |
| guidance_replication | 263 | 17 | control | brows | 15.697349 | 0.125291 | 0.227567 | 44 | 2110 |
| guidance_replication | 6681 | 17 | control | eyes | 13.676950 | 0.156595 | 0.202704 | 66 | 859 |
| guidance_replication | 6681 | 17 | control | brows | 16.561234 | 0.128675 | 0.402897 | 462 | 2306 |
| guidance_replication | 10149 | 17 | control | eyes | 12.221855 | 0.198232 | 0.032124 | 472 | 1672 |
| guidance_replication | 10149 | 17 | control | brows | 11.105102 | 0.252795 | 0.214748 | 155 | 1872 |
| guidance_replication | 263 | 17 | candidate | eyes | 11.635754 | 0.205018 | 0.208927 | 219 | 1349 |
| guidance_replication | 263 | 17 | candidate | brows | 17.714532 | 0.100512 | 0.245150 | 44 | 2110 |
| guidance_replication | 6681 | 17 | candidate | eyes | 13.651156 | 0.164669 | 0.251216 | 66 | 859 |
| guidance_replication | 6681 | 17 | candidate | brows | 18.101333 | 0.105319 | 0.443473 | 462 | 2306 |
| guidance_replication | 10149 | 17 | candidate | eyes | 13.223401 | 0.174181 | 0.082073 | 472 | 1672 |
| guidance_replication | 10149 | 17 | candidate | brows | 11.326246 | 0.245596 | 0.219165 | 155 | 1872 |
| guidance_replication | 263 | 29 | control | eyes | 10.813308 | 0.227359 | 0.215541 | 219 | 1349 |
| guidance_replication | 263 | 29 | control | brows | 14.998867 | 0.137949 | -0.074632 | 44 | 2110 |
| guidance_replication | 263 | 29 | candidate | eyes | 10.907815 | 0.229672 | 0.217050 | 219 | 1349 |
| guidance_replication | 263 | 29 | candidate | brows | 16.778239 | 0.107986 | -0.107891 | 44 | 2110 |
| guidance_replication | 6681 | 29 | control | eyes | 12.780816 | 0.182777 | 0.069960 | 66 | 859 |
| guidance_replication | 6681 | 29 | control | brows | 16.590089 | 0.127757 | 0.347414 | 462 | 2306 |
| guidance_replication | 6681 | 29 | candidate | eyes | 13.700663 | 0.166409 | 0.062797 | 66 | 859 |
| guidance_replication | 6681 | 29 | candidate | brows | 18.635092 | 0.094378 | 0.343125 | 462 | 2306 |
| guidance_replication | 10149 | 29 | control | eyes | 12.171044 | 0.198207 | 0.062316 | 472 | 1672 |
| guidance_replication | 10149 | 29 | control | brows | 11.216110 | 0.252813 | 0.216938 | 155 | 1872 |
| guidance_replication | 10149 | 29 | candidate | eyes | 12.827374 | 0.182200 | 0.087807 | 472 | 1672 |
| guidance_replication | 10149 | 29 | candidate | brows | 11.514636 | 0.243684 | 0.219316 | 155 | 1872 |
| guidance_replication | 263 | 43 | control | eyes | 8.953778 | 0.297354 | 0.062082 | 219 | 1349 |
| guidance_replication | 263 | 43 | control | brows | 15.451886 | 0.130702 | 0.206092 | 44 | 2110 |
| guidance_replication | 263 | 43 | candidate | eyes | 9.185697 | 0.290413 | 0.092299 | 219 | 1349 |
| guidance_replication | 263 | 43 | candidate | brows | 15.873802 | 0.122610 | 0.273350 | 44 | 2110 |
| guidance_replication | 6681 | 43 | control | eyes | 11.902434 | 0.206133 | 0.091562 | 66 | 859 |
| guidance_replication | 6681 | 43 | control | brows | 16.465080 | 0.130144 | 0.311435 | 462 | 2306 |
| guidance_replication | 6681 | 43 | candidate | eyes | 12.854039 | 0.184189 | 0.157693 | 66 | 859 |
| guidance_replication | 6681 | 43 | candidate | brows | 18.410249 | 0.100247 | 0.309038 | 462 | 2306 |
| guidance_replication | 10149 | 43 | control | eyes | 12.201494 | 0.194231 | 0.020095 | 472 | 1672 |
| guidance_replication | 10149 | 43 | control | brows | 11.064154 | 0.256613 | 0.214180 | 155 | 1872 |
| guidance_replication | 10149 | 43 | candidate | eyes | 12.990224 | 0.175386 | 0.046889 | 472 | 1672 |
| guidance_replication | 10149 | 43 | candidate | brows | 11.338608 | 0.248328 | 0.217929 | 155 | 1872 |
| appearance | 263 | 17 | clean | eyes | 11.501457 | 0.202997 | 0.228068 | 219 | 1349 |
| appearance | 263 | 17 | clean | brows | 14.209893 | 0.152475 | 0.195280 | 44 | 2110 |
| appearance | 6681 | 17 | clean | eyes | 13.488366 | 0.161393 | 0.219119 | 66 | 859 |
| appearance | 6681 | 17 | clean | brows | 16.897198 | 0.122243 | 0.417755 | 462 | 2306 |
| appearance | 10149 | 17 | clean | eyes | 12.199600 | 0.196919 | 0.046468 | 472 | 1672 |
| appearance | 10149 | 17 | clean | brows | 11.039601 | 0.254760 | 0.225644 | 155 | 1872 |
| interaction | 263 | 17 | clean | eyes | 12.152733 | 0.190532 | 0.262078 | 219 | 1349 |
| interaction | 263 | 17 | clean | brows | 15.818034 | 0.127062 | 0.144341 | 44 | 2110 |
| interaction | 6681 | 17 | clean | eyes | 14.001976 | 0.152775 | 0.228296 | 66 | 859 |
| interaction | 6681 | 17 | clean | brows | 18.146671 | 0.103537 | 0.445012 | 462 | 2306 |
| interaction | 10149 | 17 | clean | eyes | 12.753865 | 0.181516 | 0.047354 | 472 | 1672 |
| interaction | 10149 | 17 | clean | brows | 11.326281 | 0.245347 | 0.221931 | 155 | 1872 |

Anatomical eye SSIM from this calculation matches the existing stored scorer within 1e-12. Brow annotations and this diagnostic are supplementary, not a replacement protocol or a newly tuned loss. See all paired primary measurements in FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md.
