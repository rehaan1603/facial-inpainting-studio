# Facial Inpainting Studio — Last-minute Viva Sheet

For 6 October 2026. See the full notes for protocols and limitations.



| Remember | Answer |
|---|---|
| Main base model | Frozen ReF-LDM |
| Main proposed code | `scripts/target_region_compatibility_v1.py` |
| Main adapter | 10→16→1 MLP, 193 parameters, shared across 16 attention layers |
| Regions | Left eye, right eye, nose, mouth, context |
| Adapter dataset | 16 training identities + 4 observed development identities |
| References | Four per identity in the matched study |
| v2–v5 optimizer/budget | AdamW, learning rate 0.001, 64 updates per arm |
| Training seed | 20261002 |
| Inference seeds | 17, 29, 43 |
| Inference | 512 pixels, 50 DDIM steps, guidance 1.5 |
| Current-family training count | 8 runs, 640 combined updates across distinct arms |
| Latest loss | Diffusion MSE + timestep-scaled cropped FaceNet loss |
| Baseline clean FaceNet / LPIPS | 0.91497 / 0.27725 |
| v5 clean FaceNet / LPIPS | 0.91623 / 0.27743 |
| v5 misleading FaceNet | 0.90414 versus paired baseline 0.90932; 11 pairs |
| What is frozen? | Backbone, VQ encoder/decoder and recognition weights |
| What remains? | Consistent joint quality gains, regional rejection, broader independent validation |

**A safe closing statement:**

> “We have completed a working local application and a controlled series of training and evaluation experiments. The results show that identity and perceptual quality must be assessed together. Our next work focuses on rejecting only misleading reference regions, followed by broader validation. The code, checkpoints and measured results make each step traceable.”



Main code: [Regional compatibility adapter](<C:/Users/rehaa/OneDrive/Documents/ChatGPT/Facial inpainting Project/scripts/target_region_compatibility_v1.py:37>)
