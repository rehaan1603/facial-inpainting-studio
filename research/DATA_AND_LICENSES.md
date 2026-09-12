# Data and third-party provenance

Updated 12 September 2026 for the reference-assisted implementation. Sources linked in the new model section were checked on this date; original dataset and object-asset records retain their earlier audit scope.

Original face datasets remain in the supplied download folders. No original files were moved, edited, or uploaded. Source files and checkpoints are excluded from the code release. This file records provenance; it does not grant redistribution rights.

## Face sources

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): attributes, identity labels and official partitions linked through the supplied HQ mapping.
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ): 30,000 HQ images and parsing annotations. Follow the original research-use agreement.
- [LaPa](https://github.com/jd-opensource/lapa-dataset): 22,168 images with landmarks and labels; retain original dataset terms. Image-level partitions do not prove identity-disjoint evaluation.

All 52,168 face images were decoded and hashed. Working manifests first quarantined 224 files in 112 exact-file duplicate groups. The cross-split/cross-source pHash screen then proposed 496 pairs. All 496 were visually reviewed: 11 near-identical photograph pairs, one scene-sequence pair conservatively quarantined, and 484 different-photograph pairs. This is photographic similarity triage, not recognition of people's identities. An independent review is still appropriate for publication.

The current reviewed v2 manifests retain 29,926 HQ and 21,994 LaPa images. The additional quarantine removes 20 HQ and four LaPa files. No pilot/v2/v3 case overlaps these exclusions. Global pHash screening does not prove that every transformed duplicate was found. Cropped or flipped versions can escape it; model pretraining overlap also remains unresolved.

## Object assets

Twelve photographed object cutouts are selected deterministically from COCO val2017 instance annotations: two each of umbrella, bottle, cup, cell phone, book and teddy bear. Selection uses COCO image metadata license ID 4 (CC BY 2.0), adequate object size, non-crowd polygons and image margins. All twelve were visually inspected. Some annotated silhouettes contain separated visible parts or coarse boundaries, retained without performance-based filtering.

The [COCO terms](https://cocodataset.org/#termsofuse) distinguish annotation licensing (CC BY 4.0) from the individual images' Flickr licenses. `object_assets.json` records image URLs, Flickr source URLs, instance IDs, licenses, modifications and hashes. `coco_download.json` records the official annotation download. Photographer attribution must be completed before redistributing example composites; this local package does not publish them.

Pillow rasterizes the original polygons, crops RGBA assets and resizes them. Pixels with resized alpha at least 128 form a hard opaque silhouette. These are synthetic composites using real object photographs, not real captured facial occlusions or a newly collected real-world dataset. The objects were never used to train the local refiners. Their occurrence in backbone or evaluator pretraining is unknown.

## Checkpoints and evaluation

The LaMa TorchScript export's source and checksums are in `baseline_provenance.json`. Official ResShift source commit, model hashes, preprocessing and S-Lab noncommercial license are in `resshift_provenance.json`; original third-party source remains in the external cache. LPIPS uses the original AlexNet evaluator. `PRETRAINING_EXPOSURE_LEDGER.md` records known and unresolved memberships. None of these models establishes true identity recovery from missing pixels.

## Reference-assisted model dependencies

| Component | Local provenance | Applicable distinction |
|---|---|---|
| SDXL inpainting 0.1 | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`; verified fp16 files recorded in `osor_downloads.json` | The model card identifies the CreativeML Open RAIL++-M license. The reference CLI uses this base, not the OSOR fine-tuned weights, despite the shared cache directory name. [Official model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| FaceID Portrait SDXL adapter | `h94/IP-Adapter-FaceID`, revision `43907e6f44d079bf1a9102d9a6e56aef7a219bae`; `ip-adapter-faceid-portrait_sdxl.bin`, 749,822,515 bytes; SHA256 `5631ce7824cdafd2db37c5e85b985730a95ff59c5b4fc80c2b79b0bee5711512` | The author restricts FaceID models to noncommercial research. Download availability does not grant commercial deployment or unrestricted redistribution rights. [Author model card](https://huggingface.co/h94/IP-Adapter-FaceID) |
| InsightFace buffalo_l | Official release ZIP and individual ONNX hashes in `reference_faces_provenance.json`; ZIP SHA256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` | Library code is MIT licensed; bundled pretrained models are separately limited to noncommercial research. The CLI uses detector and face-embedding modules for supplied references. [Official library/model terms](https://github.com/deepinsight/insightface/tree/master/python-package) |

The local implementation keeps model weights in the external cache and loads reference images locally. Retain original model terms and keep these third-party weights out of the source archive. The reference adapter and generator are existing pretrained dependencies; no new local reference-model training or new license grant is implied.

Functional examples derive from the reviewed CelebAMask-HQ validation partition. `outputs/reference_examples/selection_audit.json` records selection by identity label, file hashes, and face detection before generation. Selection and detection failures are retained. Same-person labels, distinct files, and visually plausible results do not settle reference-target duplication, identity accuracy, or pretraining exposure. Do not redistribute the face photographs or generated example composites without the applicable permissions and attribution review.

Boundary harmonization uses established Poisson image editing through OpenCV and does not change the provenance or restrictions of its image inputs. The raw generator result and compositing mode must remain distinguishable in a research comparison. The local environment record is `reference-environment-lock.txt`.

Local project code has no selected public redistribution license yet. An author must choose one, verify third-party compatibility, supply their author information, and review artifact permissions before any public release. Nothing has been submitted or published.
