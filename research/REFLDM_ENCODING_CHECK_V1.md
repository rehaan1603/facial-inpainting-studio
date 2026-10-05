# Official-input reference encoding check

5 October 2026. One previously observed development identity (6681), official target 11491.png, seed 17, 100 DDIM steps, five references, float32. Only reference encoding changes. This is a diagnostic, not a reproduced paper benchmark.

| Encoding | PSNR | SSIM | LPIPS | FaceNet | Peak tensor GiB |
|---|---:|---:|---:|---:|---:|
| author_rgb_encoding | 22.941815 | 0.550950 | 0.240699 | 0.896407 | 7.09 |
| separate_encoding | 22.879941 | 0.550373 | 0.241143 | 0.871496 | 2.38 |

The author-style concatenated encoding improves all four available quality/identity metrics on this one case. Separate encoding reduces memory. This identifies a relevant implementation difference; it does not establish a general improvement. ArcFace is unavailable because the target detector finds no face. No detector thresholds were changed.

The history audit scanned 10,323 JSON files with no parse errors. Broad metadata mentions cover all official rows, including inventories/exclusions; they do not prove image exposure. Explicit completed/metric/training-trace records identify 32 identity labels touching 32 benchmark rows. This scan does not yet resolve all training-list, CSV, log or external-history exposure.

Next: compare both encodings across the remaining already-observed official development cases before considering a production change. Preserve official mappings and clearly separate any filtered evaluation subset.