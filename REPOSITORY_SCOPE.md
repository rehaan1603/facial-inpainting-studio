# Repository checkpoint

This private repository preserves the implementation and research record as of 12 September 2026. It includes the local browser application, model integration and evaluation scripts, configuration examples, environment locks, research protocols and measured-result reports.

Datasets, uploaded faces, reference photos, generated face images, model weights, Python environments, local path configuration, raw experiment outputs and delivery ZIP archives stay on the original laptop. The repository therefore does not run inference from a fresh clone until the documented environments, permitted models, datasets and local configuration are restored. The trained small-refiner checkpoints also remain local.

Start the configured application on the original laptop with Start Studio.cmd and open http://127.0.0.1:8765/. Keep that server running while using the website. The GPU service binds only to 127.0.0.1; this is not an internet-hosted app or a GitHub Pages deployment.

The reference-photo feature uses existing pretrained methods. Earlier measured experiments, unfinished comparisons and publication requirements are distinguished in research/PROJECT_STATUS.md. Research results are not claims of accepted publication or verified hidden-face recovery. Third-party data, code and weights retain their own terms; see research/DATA_AND_LICENSES.md.
