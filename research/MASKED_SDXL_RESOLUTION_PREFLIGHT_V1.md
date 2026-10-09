# Single-change resolution diagnostic

8 October 2026; fixed before 1024 generation. The initial 512 SDXL eye output fills the hole but has incorrect gaze and boundary appearance. Test whether the existing 1024 path improves facial structure without changing references or losses.

Eight frozen cases, clean references, seed17; include reference-detection failures rather than replacing photos. Change only `model_resolution` 512→1024. The existing runner consequently resizes the same image/mask and enables VAE tiling at1024; this is an existing implementation consequence, not a separate tuned intervention. Preserve steps30, guidance5, FaceID scale0.8, strength1, prompt, reference ordering, DDIM scheduler and hard output composition at512.

Save in a new directory `outputs/masked_sdxl_resolution_v1`. No training, checkpoint modifications or deployment. Compare only jointly completed and jointly valid 512/1024 rows, including coverage. Evaluate hole/eye crop LPIPS, hole PSNR/SSIM, FaceNet and ArcFace, visible-pixel preservation, timing, memory and visual boundary/gaze differences. ArcFace is a conditioning metric here.

Pilot success requires improved hole LPIPS with no worsening of hole PSNR/SSIM and identity metrics, plus plausible reconstructed structure. A win is only a development indication: additional seeds and corrupted references are required before promotion. Do not choose another seed, reference subset or loss after seeing the results and label it this experiment.
