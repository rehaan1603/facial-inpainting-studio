# Reference appearance addition: fixed three-person pilot

9 October 2026. Freeze before appearance inference. Guidance-only and scheduler-only diagnostics improve some measures but do not reconstruct matching iris/gaze/brow details. The existing generator passes normalized InsightFace recognition embeddings. An appearance input is a technically justified next intervention.

## Single mechanism change

Retain SDXL inpainting and the existing multi-reference FaceID Portrait branch at scale **0.8**. Add the author's pretrained **IP-Adapter Plus Face SDXL ViT-H** visual-reference branch at fixed scale **0.2**. This is an additional pretrained baseline component, not a newly trained or novel mechanism. Use the SAME four real photos, same order, for identity and appearance. No target-derived descriptions, synthetic references or target pixels enter conditioning. All other settings match the saved control: persons 263, 6681, 10149; eyes only; clean references; seed 17; DDIM 30; CFG 5; strength 1; model 1024 / output 512; unchanged Poisson finish and mask.

Align each supplied reference to a 224-pixel face crop using its own detected five landmarks. Compute CLIP ViT-H penultimate hidden states on CPU before loading the diffusion model; cache four independent reference features and zero-input unconditional features. Preserve reference order and separate tokens. Conditioning is not spatial eye-pixel correspondence: reference pose/expression can still influence reconstruction incorrectly. No claim of exact hidden-region recovery.

Assets are pinned to official `h94/IP-Adapter` revision `018e402774aeeddd60609b4ecdb7e298259dc729`. Verify encoder (2,528,373,448 bytes) and adapter (847,517,512 bytes) SHA-256 against pinned LFS metadata; verify config Git blob. Download only into a new model-cache folder. Preserve the existing Portrait weights. Do not redistribute models or photos with source code.

## Wiring and equivalence prerequisite

An experimental wrapper appends the appearance embeddings and maintains both attention branches. The existing generator remains untouched. Before the candidate calls, set appearance scale to ZERO and replay person 263. Require both raw and final images to be byte-exact with the original CFG 5 control. Record projection classes, embedding/token shapes, scales, input/reference/asset/source hashes, timing and memory. If equivalence fails, stop: debug wiring before generating candidates. No large training and no production change.

## Evaluation and decision

Generate three candidates at the fixed appearance scale, saving raw/hard/final images and failures. Use the unchanged masked evaluator and the same three control images. Primary: hole SSIM, hole PSNR, hole crop LPIPS, anatomical-eye contextual LPIPS, FaceNet and ArcFace with coverage; inspect iris colour, gaze and brows on every comparison. Visible pixels must remain exact. Do not select different scales or seeds for different people. Any mean or meaningful per-person identity regression blocks universal promotion. Passing this small screen motivates separate multiseed and corrupted-reference validation; it does not establish population accuracy.

Official sources checked on 9 October 2026: [IP-Adapter model repository](https://huggingface.co/h94/IP-Adapter), [Diffusers adapter documentation](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter). The documentation distinguishes visual-detail Plus conditioning from identity FaceID conditioning. The proposal is an inference from that distinction and observed errors, not a published guarantee of iris preservation.
