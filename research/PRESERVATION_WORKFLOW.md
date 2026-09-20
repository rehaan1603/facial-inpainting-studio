# Using the evidence map with a saved reconstruction

The confidence editor at http://127.0.0.1:8765/confidence.html exports three matching PNGs: observed image, binary damage mask and grayscale confidence map. White confidence keeps observed pixels, black uses reconstruction, intermediate values blend them. This is experimental evidence weighting, not verified blind restoration.

For an existing 512-pixel reference reconstruction, apply the edited confidence map without loading a GPU model:

```powershell
.\.venv\Scripts\python.exe scripts\apply_saved_preservation.py --image "original-input.png" --mask "original-mask.png" --confidence "evidence-confidence.png" --generated "saved-result.png" --metadata "saved-result.json" --output "new-preserved-result.png"
```

Use the original input and mask from that generation. Their file hashes, the saved result hash and its metadata must match. Browser re-encoding an input may change its file hash even when pixels look identical, so keep the original generation files. The confidence-map dimensions must match the oriented input. Do not expand the damage mask beyond the region that was actually reconstructed. All outputs need new paths; prior results and inputs are protected.

A real saved development reconstruction was processed successfully. Output pixels outside damage remain exactly unchanged; overwrite and mismatched-generation rejection were checked. See saved_preservation_check_v1.json. This operation does not generate a new face, repair errors in the saved reconstruction, or resolve the blocked inference runtime.

## Current runtime blocker

Windows Code Integrity event 3077 reports that scipy/linalg/_batched_linalg.cp312-win_amd64.pyd and scipy/sparse/csgraph/_matching.cp312-win_amd64.pyd fail the device application-control policy. A later single-module import succeeded, but full model loading still failed. No dependency or security policy was changed. Repeated generation retries are not a repair.

The device administrator/owner must review the blocked Python package under the existing application-control policy and provide a permitted runtime or an explicitly approved policy decision. Do not disable Smart App Control, rename/move blocked libraries to evade detection, or claim the runtime is repaired from a partial import. After resolution, repeat a complete reference reconstruction in a new output folder before running more experiments.
