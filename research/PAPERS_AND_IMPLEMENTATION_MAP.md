# Papers and implementation map

Checked 30 September 2026. This project combines pretrained methods; it is not yet a validated extension of one base paper. Running author weights is not reproduction of published benchmark scores. Our mask/evidence experiments have not established novelty.

## Which is the base paper?

For the **single-image facial inpainting branch**, the most accurate implementation foundation is **Efficient Diffusion Model for Image Restoration by Residual Shifting (2024)**, the extended ResShift work including inpainting. LaMa is a separate comparator. Do not cite only the original super-resolution ResShift paper as if it describes the full inpainting implementation.

For **reference-based blur/noise restoration**, the base is **ReF-LDM (NeurIPS 2024)**. It is a restoration model, not a verified solution for completely erased facial components.

For the **website's reference-guided missing-region mode**, the implementation combines **SDXL inpainting and IP-Adapter FaceID Portrait**. FaceID Portrait is a released adapter variant; the original IP-Adapter paper is not itself a full specification of that later checkpoint.

If a project form requires one base paper for the current reference-restoration direction, use ReF-LDM and state that the project also investigates missing-region completion using separate baselines. If the title promises single-image completion, use the extended ResShift paper instead. Neither choice establishes a new contribution.

## Implemented generators and experimental candidates

| Paper / implementation | Role and status | Local implementation evidence |
|---|---|---|
| [Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa), WACV 2022](https://arxiv.org/abs/2109.07161) | Implemented single-image baseline; face quality remains limited. | scripts/inpaint.py; Big-LaMa checkpoint |
| [Efficient Diffusion Model for Image Restoration by Residual Shifting, 2024](https://arxiv.org/abs/2403.07319) | Implemented ResShift face-inpainting baseline. | scripts/resshift_adapter.py; inpaint_lama256_face configuration |
| [ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, 2023](https://arxiv.org/abs/2307.12348) | Earlier foundation; distinguish super-resolution from the extended restoration work. | Same author repository lineage |
| [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, 2023](https://arxiv.org/abs/2307.01952) | Implemented pretrained inpainting backbone. | scripts/reference_inpaint.py |
| [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models, 2023](https://arxiv.org/abs/2308.06721), [FaceID Portrait release](https://huggingface.co/h94/IP-Adapter-FaceID) | Implemented identity conditioning using the Portrait SDXL checkpoint, not a newly trained adapter. | scripts/reference_inpaint.py; scripts/studio_reference_engine.py |
| [ReF-LDM: A Latent Diffusion Model for Reference-based Face Image Restoration, NeurIPS 2024](https://arxiv.org/abs/2412.05043), [author code](https://github.com/ChiWeiHsiao/ref-ldm) | Implemented blur/noise restoration; also tested as a refinement stage. | scripts/refldm_restore.py; scripts/studio_refldm_restore.py |
| [Reference-Guided Large-Scale Face Inpainting with Identity and Texture Control, 2023 preprint](https://arxiv.org/abs/2303.07014), [author code](https://github.com/WuyangLuo/RefFaceInpainting) | Implemented and evaluated research comparator. Failed our quality gate; not a website default. | src/preservation/refface_baseline.py; research/REFFACE_BASELINE_REVIEW_V1.md |
| [RAD: Region-Aware Diffusion Models for Image Inpainting, CVPR 2025](https://github.com/srk1995/RAD) | Newly downloaded official FFHQ checkpoint and isolated runtime. Strict parameter loading passed; initial smoke output is unusable. Compatibility/reproduction is unverified; not deployed. | outputs/rad_smoke_v1; isolated rad_env_v1 and rad_source_v1 |

## Evaluation papers

These metrics evaluate outputs; they do not reconstruct the face. A good metric score alone does not establish correct identity, gaze or expression.

- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html): identity embeddings and similarity. Because ArcFace also conditions some generators, its score is not fully independent of conditioning.
- [FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015](https://arxiv.org/abs/1503.03832): additional identity-similarity evaluation. The implementation/checkpoint must be cited alongside the paper; embeddings are not an accuracy percentage.
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018](https://arxiv.org/abs/1801.03924): LPIPS perceptual distance.
- [Making a Completely Blind Image Quality Analyzer (NIQE)](https://live.ece.utexas.edu/research/quality/niqe_spl.pdf): no-reference natural-image quality measure.
- **No-Reference Image Quality Assessment in the Spatial Domain (BRISQUE), IEEE Transactions on Image Processing, 2012**: no-reference quality. [Author laboratory's quality-assessment resources](https://live.ece.utexas.edu/research/Quality/nrqa.htm).
- **Image Quality Assessment: From Error Visibility to Structural Similarity, IEEE Transactions on Image Processing, 2004**: SSIM. [Author laboratory's algorithm catalogue](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm).
- PSNR and masked MAE are conventional reconstruction measures; report formula, scale, mask region and aggregation instead of attributing them to a new model paper.

## Related work, not implemented merely by citing it

- [MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://arxiv.org/abs/2203.15270): background for mask-aware inpainting; our compact mask refiner is not MAT.
- [PATMAT: Person Aware Tuning of Mask-Aware Transformer for Face Inpainting, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Motamed_PATMAT_Person_Aware_Tuning_of_Mask-Aware_Transformer_for_Face_Inpainting_ICCV_2023_paper.pdf): personalization precedent; not implemented here.
- [Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Xu_Personalized_Face_Inpainting_With_Diffusion_Models_by_Parallel_Visual_Attention_WACV_2024_paper.pdf): reference-conditioned inpainting precedent; not implemented here.
- [When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions, 2026 preprint](https://arxiv.org/abs/2608.04820): recent semantic-prior related work; not an implemented or validated component.

For the wider novelty comparison, also consult STRUCTURAL_PRIOR_ART_20260928.md and PUBLICATION_AUDIT.md. Dataset publications and license terms must be cited separately in the final manuscript. Synthetic references derived from the clean target are demo material, not independent identity evidence.
