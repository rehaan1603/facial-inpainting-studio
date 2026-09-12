# Inpainting Studio

Local browser interface connected to the existing GPU inference implementation. The application is served by `server.py`; `dist` contains its authored static frontend. It requires the parent project's CUDA Python environment, model cache and `configs/local.json`. The static files alone cannot perform inpainting or be deployed as a functioning GPU application.

Double-click `Start Studio.cmd` in the project root, or run `& ./webapp/start.ps1` from PowerShell. Open http://127.0.0.1:8765 and leave the server window open. If the app is already running, use that page instead of starting a second copy. Ctrl+C in the server window stops it. To use another port, run `.venv/Scripts/python.exe webapp/server.py --port 8766`.

1. Upload a PNG, JPEG or WebP face image, or load the available research sample.
2. Select the square crop or whole-image fit. Framing resets the mask; the displayed frame is the exact input used for inference.
3. Paint the replacement area. Brush, eraser, undo, clear and mask visibility are available. A PNG mask can be uploaded instead; white selects pixels, black/transparent retains them. Uploaded masks are fitted to the displayed square, so supply a mask aligned to that frame.
4. Select LaMa, ResShift or Use 3–4 reference photos and a mask treatment, then reconstruct. In reference mode add three or four clear photos of the same person, one face per image. Load research sample in this mode loads matching references. Match colour at edges enables optional Poisson harmonization; uncheck for hard composition. LaMa is the default fast control. The experimental learned control uses the original generic seed-17 refiner with its recorded 0.5 threshold; it is not presented as a proven improvement.
5. Move Before / after to inspect the reconstruction. Download the result and effective mask.

Outputs are 256 × 256 PNGs for LaMa/ResShift and 512 × 512 PNGs for the reference mode. All modes preserve the resized input exactly outside the effective mask. Expansion or refinement can change which pixels are considered missing. This interface neither identifies people nor verifies the true appearance of hidden facial features. It is a research demo, and does not itself establish publication novelty.

Each run is retained locally under `outputs/webapp_runs/<run-id>/` with input, supplied/effective masks, result and JSON settings. Reference runs also retain the supplied references, raw generated image and hard-composed image for local inspection. Uploaded faces are not included in delivery archives. The server binds only to this laptop's loopback interface, performs one inference at a time, and makes no external image-upload requests. Avoid running another large GPU job at the same time; model loading and concurrent workloads affect latency. The phone layout is supported, but the loopback URL is accessible only on the laptop.

Validation: ten HTTP boundary tests in `scripts/test_webapp.py`; five real HTTP/GPU/PNG checks in `research/webapp_inference_checks.json`; successful browser LaMa/ResShift flows; desktop and narrow-layout checks. Functional checks are not quality metrics. Optional WebMCP exposes `get_inpainting_state` and `run_local_inpainting` using the same visible state and action.

Reference validation: `research/reference_webapp_check.json` records a four-photo upload, GPU reconstruction, both download events and exact outside-mask pixel preservation. Nine boundary-blending tests pass in the separate reference environment. WebMCP reconstruction acknowledges immediately; read its state tool for progress and completion. No reference quality benchmark is complete yet.
