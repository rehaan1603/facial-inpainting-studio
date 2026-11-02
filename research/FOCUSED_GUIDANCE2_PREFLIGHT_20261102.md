# Frozen CFG2 development screen — 2 November 2026

Hypothesis: reducing text classifier-free guidance from 3 to 2 reduces text-driven appearance drift while retaining FaceID reference influence. Prior CFG5-to3 replication improved average regional structure; spatial-capacity training did not surpass CFG3. This is an inference calibration experiment, not a novel mechanism or training run.

## Fixed design
Only guidance changes. Three existing development people 263, 6681, 10149; eyes; all four original real references; original masks and damaged inputs; seed17; 30DDIM steps; FaceID0.8; strength1; 1024 inference/512 output; Poisson finishing. No appearance adapter, scorer or spatial adapter. No new dataset or checkpoint. GPU runtime about five minutes for four calls. No hidden target pixels in generation.

Preserve existing generator and metric implementations. Fresh output directory outputs/focused_eye_guidance2_20261102; refuse overwrite. Before candidates, CFG3 raw/final replay for263 must be pixel-exact against existing CFG3. Pin script/protocol/input/reference hashes in config; retain every failed and completed attempt. Do not tune parameters per person or select favorable seeds.

## Evaluation and acceptance
Use unchanged score_masked_sdxl_v1.py. All three valid pairs required. Mean holeSSIM must strictly improve; wholePSNR, wholeSSIM, holePSNR, FaceNet and ArcFace must not decrease; whole/hole/eyeLPIPS must not increase. Each person must retain FaceNet and ArcFace within0.005 of control. Visible pixels must remain exact. Inspect every raw/final reconstruction and eye crop, including iris/brow/gaze errors. Record all failed gates. A numerical pass alone never replaces website defaults.

If gates pass, replicate the same frozen CFG2 vsCFG3 comparison at seeds29 and43, using existing controls, before considering any integration. If they fail, stop this recipe and retain CFG3. No repeated parameter search in this screen. All three are reused development cases, not unknown-client accuracy estimates. Reserved final identities remain untouched.
